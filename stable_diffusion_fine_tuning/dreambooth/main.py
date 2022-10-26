import os
from io import BytesIO
import math
from typing import List, Dict, NamedTuple

import requests
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
import wandb
import numpy as np
from PIL import Image

from stable_diffusion_fine_tuning.dreambooth.dataset import ClassDataset, PromptDataset, create_dreambooth_dataloader
from stable_diffusion_fine_tuning.config import AdamConfig, GradientConfig


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DreamboothConfig(NamedTuple):
    name: str
    instance_images: List[str]
    instance_prompt: str
    class_prompt: str
    eval_prompt: str


def run(
    pipeline,
    configs: List[DreamboothConfig],
    root_dir: str,
    resolution: int = 256,
    center_crop: bool = True,
    max_train_steps: int = 1100,
    train_batch_size: int = 1,
    inference_batch_size: int = 2,
    max_grad_norm: float = 1.0,
    seed: int = 3434554,
    with_prior_preservation: bool = True,
    num_class_images: int = 12,
    prior_loss_weight: float = 0.5,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    log_interval: int = 10,
    eval_interval: int = 100,
    gradient_config: GradientConfig = GradientConfig(grad_acc=1, gradient_checkpointing=False),
    adam_config: AdamConfig = AdamConfig(lr=2e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
    eval_prompts: Dict[str, str] = None,
    device = "cuda:0"
):
    class_datasets = []
    for config in configs:
        save_dir = os.path.join(root_dir, config.name)
        _download_images(config.name, config.instance_images, save_dir)
        instance_images = [
            Image.open(os.path.join(save_dir, filename)).convert("RGB")
            for filename in os.listdir(save_dir)
        ]
        
        if with_prior_preservation:
            class_data_root_dir = os.path.join(root_dir, f"{config.name}_prior")
            with torch.no_grad():
                _generate_class_images(
                    pipeline, config.name, num_class_images, config.class_prompt,
                    inference_batch_size, class_data_root_dir
                )
                class_images = [
                    Image.open(os.path.join(class_data_root_dir, filename)).convert("RGB")
                    for filename in os.listdir(class_data_root_dir)
                ]
                torch.cuda.empty_cache()
        
        class_datasets.append(
            ClassDataset(
                instance_images=instance_images,
                instance_prompt=config.instance_prompt,
                class_images=class_images,
                class_prompt=config.class_prompt
            )
        )

    return train(
        pipeline=pipeline,
        class_datasets=class_datasets,
        resolution=resolution,
        center_crop=center_crop,
        max_train_steps=max_train_steps,
        train_batch_size=train_batch_size,
        max_grad_norm=max_grad_norm,
        seed=seed,
        with_prior_preservation=with_prior_preservation, 
        prior_loss_weight=prior_loss_weight,
        lr_scheduler=lr_scheduler,
        lr_warmup_steps=lr_warmup_steps,
        log_interval=log_interval,
        eval_interval=eval_interval,
        gradient_config=gradient_config,
        adam_config=adam_config,
        eval_prompts=eval_prompts,
        device=device
    )


def train(
    pipeline,
    class_datasets: List[ClassDataset],
    resolution: int = 256,
    center_crop: bool = True,
    max_train_steps: int = 1100,
    train_batch_size: int = 1,
    max_grad_norm: float = 1.0,
    seed: int = 3434554,
    with_prior_preservation: bool = True, 
    prior_loss_weight: float = 0.5,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    log_interval: int = 10,
    eval_interval: int = 100,
    gradient_config: GradientConfig = GradientConfig(grad_acc=1, gradient_checkpointing=False),
    adam_config: AdamConfig = AdamConfig(lr=2e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2),
    eval_prompts: Dict[str, str] = None,
    device = "cuda:0"
):
    if gradient_config.grad_acc != 1:
        raise ValueError("Gradient accumulation is not supported yet")

    if not wandb.run:
        wandb_config_keys = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
        if all([key in os.environ for key in wandb_config_keys]):
            wandb.init(dir=".")
        else:
            if eval_prompts:
                raise ValueError(
                    "eval_prompts is not None so model evaluation is activated."
                    "No wandb run in progress."
                    f"We can start a run for you but you need to set the following environment variables: {wandb_config_keys}"
                )
    
    if wandb.run:
        wandb.config.update({
            "resolution": resolution,
            "center_crop": center_crop,
            "max_train_steps": max_train_steps,
            "train_batch_size": train_batch_size,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "with_prior_preservation": with_prior_preservation,
            "prior_loss_weight": prior_loss_weight,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "gradient_config": gradient_config,
            "adam_config": adam_config
        })


    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    unet = pipeline.unet.to(device)
    text_encoder = pipeline.text_encoder.to(device)
    tokenizer = pipeline.tokenizer
    vae = pipeline.vae.to(device)

    if gradient_config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    params_to_optimize = list(unet.parameters()) + list(text_encoder.parameters())
        
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=adam_config.lr,
        betas=adam_config.betas,
        weight_decay=adam_config.weight_decay,
        eps=adam_config.eps
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

    train_dataloader = create_dreambooth_dataloader(
        class_datasets=class_datasets,
        train_batch_size=train_batch_size,
        tokenizer=tokenizer,
        image_resolution=resolution,
        center_crop=center_crop,
        with_prior_preservation=with_prior_preservation
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_config.grad_acc,
        num_training_steps=max_train_steps * gradient_config.grad_acc,
    )

    print("***** Running training *****")
    print(f"  Gradient Accumulation steps = {gradient_config.grad_acc}")
    print(f"  Total optimization steps = {max_train_steps}")
   
    global_step = 0
    loss_avg = AverageMeter()
    unet.train()
    text_encoder.train()
    for step, batch in enumerate(train_dataloader):
        print(f"step: {step}")
        # Convert images to latent space
        with torch.no_grad():
            latent_dist = vae.encode(batch["pixel_values"].to(device, non_blocking=True)).latent_dist
            latents = latent_dist.sample() * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        with torch.no_grad():
            encoder_hidden_states = text_encoder(batch["input_ids"].to(device, non_blocking=True))[0]

        # Predict the noise residual
        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if with_prior_preservation:
            # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
            noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
            noise, noise_prior = torch.chunk(noise, 2, dim=0)

            # Compute instance loss
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

            # Compute prior loss
            prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()

            # Add the prior loss to the instance loss.
            loss = loss + prior_loss_weight * prior_loss
        else:
            loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        loss_avg.update(loss.detach_(), bsz)

        if not global_step % log_interval:
            logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
            print(logs)
            if wandb.run:
                wandb.log(logs)
            
        if not global_step % eval_interval and eval_prompts:
            unet.eval()
            text_encoder.eval()
            for eval_name, eval_prompt in eval_prompts.items():
                images = _generate_images(pipeline, eval_prompt, batch_size=1, num_samples=4)
                wandb.log({f"eval_images_{eval_name}_{global_step}": wandb.Image(images)})
            unet.train()
            text_encoder.train()

        global_step += 1

        if global_step >= max_train_steps:
            break

    return unet, text_encoder


def _download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def _download_images(name: str, image_urls: List[str], save_dir: str):
    if os.path.exists(save_dir):
        print(f"Image already exist in {save_dir}; Skip!")
        return
    print(f"Downloading images of {name} ...")
    os.makedirs(save_dir)
    images = list(filter(None, [_download_image(url) for url in image_urls]))
    print(f"{len(images)} successfully downloaded")
    for i, image in enumerate(images):
        image.save(os.path.join(save_dir, f"{i}.jpg"))


def _generate_class_images(pipe, name, num_class_images: int, class_prompt: str, batch_size: int, class_data_root: str):
    if os.path.exists(class_data_root):
        print(f"Class images already exist for {name} in {class_data_root}; Skip!")
        return
    print(f"Downloading class images of {name} ...")
    os.makedirs(class_data_root)

    sample_dataset = PromptDataset(class_prompt, num_class_images)
    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size)

    for example in tqdm(sample_dataloader, desc="Generating class images"):
        images = pipe(example["prompt"]).images

        for i, image in enumerate(images):
            image.save(os.path.join(class_data_root, f"{example['index'][i]}.jpg"))


def _generate_images(pipe, prompt, batch_size=1, num_samples=4, num_inference_steps=50, guidance_scale=7.5, n_cols=4):
    with torch.no_grad():
        all_images = []
        n_batches = int(math.ceil(num_samples / batch_size))
        done = 0
        for _ in range(n_batches):
            batch_size_ = batch_size
            if (done + batch_size_) > num_samples:
                batch_size_ = num_samples - done
            images = pipe(
                prompt, num_images_per_prompt=batch_size_, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images
            all_images.extend(images)

        n_rows = int(math.ceil(num_samples / n_cols))
        return _image_grid(all_images, n_rows, n_cols)


def _image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid