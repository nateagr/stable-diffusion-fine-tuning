
import os
import shutil
import random
from argparse import Namespace
import shutil
from typing import NamedTuple
from enum import Enum
from getpass import getuser

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from transformers import get_scheduler
from transformers import set_seed
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from stable_diffusion_fine_tuning.dataset import PandasDataset


class AdamConfig(NamedTuple):
    lr = 2e-4
    betas = (0.9, 0.99)
    eps = 1e-8
    weight_decay = 1e-5


class DatasetConfig(NamedTuple):
    path = "/home/username/"
    training_ds_ratio = 0.75
    train_batch_size = 1
    test_batch_size = 1


class GradientConfig(NamedTuple):
    grad_acc: int = 1
    gradient_checkpointing = False
    enable_grad = lambda _, x: "norm" in x or "bias" in x or "emb" in x or "attn" in x


class Precision(Enum):
    FP32 = 1
    FP16 = 2
    AMP = 3


def run(
    adam_config: AdamConfig = AdamConfig(),
    dataset_config: DatasetConfig = DatasetConfig(),
    gradient_config: GradientConfig = GradientConfig(),
    seed: int = 23906,
    device: str = "cuda:0",
    train_steps: int = 600,
    warmup: int = 48,
    nb_timesteps = 1000,
    ema_decay = 0.0,
    scheduler_type = "linear",
    precision: Precision = Precision.FP32
):
    dataset_df = pd.read_pickle(dataset_config.path)
    dataset = PandasDataset(dataset_df)
    training_ds_size = int(len(dataset) * dataset_config.training_ds_ratio)
    train_ds, test_ds = torch.utils.data.random_split(dataset, [training_ds_size, len(dataset) - training_ds_size])
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=dataset_config.train_batch_size, shuffle=True, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size=dataset_config.test_batch_size, shuffle=True, num_workers=0)

    return run_with_datasets(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        adam_config=adam_config,
        dataset_config=dataset_config,
        gradient_config=gradient_config,
        seed=seed,
        device=device,
        train_steps=train_steps,
        warmup=warmup,
        nb_timesteps=nb_timesteps,
        ema_decay=ema_decay,
        scheduler_type=scheduler_type,
        precision=precision
    )


def run_with_datasets(
    train_dataloader,
    test_dataloader,
    adam_config: AdamConfig = AdamConfig(),
    gradient_config: GradientConfig = GradientConfig(),
    seed: int = 23906,
    device: str = "cuda:0",
    train_steps: int = 600,
    warmup: int = 48,
    nb_timesteps = 1000,
    ema_decay = 0.0,
    scheduler_type = "linear",
    precision: Precision = Precision.FP32,
    test_nb_samples: int = 1,
    nb_steps_between_validation: int = 200,
    test_root_dir: str = f"/home/{getuser()}/stable-diffusion-tests",
    nb_steps_between_ckpt: int = 200,
    ckpt_root_dir: str = f"/home/{getuser()}/stable-diffusion-checkpoints",
):
    if seed == 0:
        seed = random.getrandbits(16)
    print("Using seed:", seed)
    set_seed(seed)

    pipe = load_model(precision, device)
    pipe.guidance_scale = 1.0
    pipe.scheduler.set_timesteps(nb_timesteps)

    params_to_optimize =configure_gradient(pipe, gradient_config.gradient_checkpointing, gradient_config.enable_grad)

    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=adam_config.lr, weight_decay=adam_config.weight_decay,
        betas=adam_config.betas, eps=adam_config.eps
    )
    
    ema = configureExponentialMovingAverage(ema_decay, params_to_optimize)

    scheduler = get_scheduler(
        scheduler_type, optimizer, warmup, train_steps // gradient_config.grad_acc
    )

    return train(
        pipe=pipe,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        test_nb_samples=test_nb_samples,
        test_root_dir=test_root_dir,
        train_steps=train_steps,
        params_to_optimize=params_to_optimize,
        grad_acc=gradient_config.grad_acc,
        precision=precision,
        device=device,
        ckpt_root_dir=ckpt_root_dir,
        nb_steps_between_ckpt=nb_steps_between_ckpt,
        nb_steps_between_validation=nb_steps_between_validation
    )


def train(
    pipe,
    optimizer,
    scheduler,
    ema,
    train_dataloader,
    test_dataloader,
    test_nb_samples,
    test_root_dir: str,
    train_steps: int,
    params_to_optimize,
    grad_acc,
    precision: Precision,
    device: str,
    ckpt_root_dir: str,
    nb_steps_between_ckpt: int,
    nb_steps_between_validation: int
):
    losses = []
    global_i = 0
    bar = tqdm(initial=global_i, total=train_steps)
    scaler = torch.cuda.amp.GradScaler()
    dtype = torch.FloatTensor if precision in [Precision.FP32, Precision.AMP] else torch.HalfTensor

    while True:
        total_loss = 0.0
        for i, (text, image) in enumerate(train_dataloader):
            pipe.unet.train()
            torch.cuda.empty_cache()
            
            if precision == Precision.AMP:
                with torch.cuda.amp.autocast():
                    loss = get_loss(text=text, image=image, pipe=pipe, device=device, dtype=dtype)
            else:
                loss = get_loss(text=text, image=image, pipe=pipe, device=device, dtype=dtype)
            losses.append(loss.item())
            total_loss += loss.item()
            bar.set_postfix(loss=total_loss / (i + 1))
            
            if precision == Precision.AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if i % grad_acc == grad_acc - 1:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, 0.5)
                if precision == Precision.AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                ema.update()
                bar.update()
                optimizer.zero_grad()

                if global_i % nb_steps_between_ckpt == nb_steps_between_ckpt - 1:
                    save_checkpoint(pipe=pipe, optimizer=optimizer, ckpt_root_dir=ckpt_root_dir, global_i=global_i)

                if global_i % nb_steps_between_validation == nb_steps_between_validation - 1:
                    validate(
                        pipe=pipe, ema=ema, root_dir=test_root_dir, global_i=global_i,
                        nb_samples=test_nb_samples, dataloader=test_dataloader)
                
                global_i += 1
            
            if global_i >= train_steps:
                break
        
        if global_i >= train_steps:
            break

    return pipe


def load_model(precision: Precision, device: str, **sd_kwargs):
    if precision == Precision.FP16 and "revision" not in sd_kwargs:
        sd_kwargs["revision"] = "fp16"
    dtype = torch.float16 if precision == Precision.FP16 else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        use_auth_token=True,
        **sd_kwargs
    )
    pipe = pipe.to(device)
    return pipe


def configure_gradient(pipe, gradient_checkpointing: bool, enable_grad):
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    if gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    print(f"Gradient checkpointing enabled: {pipe.unet.is_gradient_checkpointing}")

    params_to_optimize = [y for x, y in pipe.unet.named_parameters() if enable_grad(x)]
    for param in params_to_optimize:
        param.requires_grad_(True)

    return params_to_optimize


def configureExponentialMovingAverage(ema_decay, params_to_optimize):
    if ema_decay > 0:
        ema = ExponentialMovingAverage(params_to_optimize, decay=ema_decay)
    else:
        ema = Namespace(
            average_parameters=lambda: open("/tmp/dontusethisplease", "w"),
            update=lambda: None
        )
    return ema


def get_loss(text, image, pipe, device, dtype):
    assert len(text) == len(image)
    timesteps = torch.LongTensor(pipe.scheduler.timesteps)
    batch_size = len(text)
    ind = torch.randint(0, len(timesteps), (batch_size,)).long()
    t = timesteps[ind]
    tokens = pipe.tokenizer(text, padding=True)
    with torch.inference_mode():
        latent_model_input = pipe.vae.encode(image.type(dtype).to(device)).latent_dist.sample().cpu() * 0.18215
        noise = torch.randn_like(latent_model_input)
        noised = pipe.scheduler.add_noise(latent_model_input, noise, t - 1)
    text_embeddings = pipe.text_encoder(
        torch.LongTensor(tokens.input_ids).to(device),
        attention_mask=torch.tensor(tokens.attention_mask).type(dtype).to(device)
    )[0]
    eps = pipe.unet(noised.to(device), t.long().to(device), encoder_hidden_states=text_embeddings)["sample"]
    loss = torch.nn.functional.mse_loss(noise.to(device), eps, reduction="none").mean([1, 2, 3]).mean()
    return loss


def get_unet_fine_tuned_state(pipe):
    return {k: v for k, v in pipe.unet.named_parameters() if v.requires_grad}


def save_checkpoint(pipe, optimizer, ckpt_root_dir: str, global_i: int):
    os.makedirs(ckpt_root_dir, exist_ok=True)
    filename = os.path.join(ckpt_root_dir, f"{global_i:06}.ckpt")
    state = {
        "step": global_i + 1,
        "model": get_unet_fine_tuned_state(pipe),
        "optimizer": optimizer.state_dict()
    }
    torch.save(state, filename)
    shutil.copy(filename, os.path.join(ckpt_root_dir, "last.ckpt"))


def validate(pipe, ema, root_dir: str, global_i: int, nb_samples, dataloader):
    validation_dir = os.path.join(root_dir, f"step_{global_i}")
    os.makedirs(validation_dir, exist_ok=True)
    pipe.unet.eval()
    with torch.inference_mode(), torch.autocast("cuda"), \
            ema.average_parameters():
        for i, (text, image) in zip(range(nb_samples), dataloader):
            with open(os.path.join(validation_dir, f"sample_{i}_text"), "wb") as fd:
                fd.write(text[0].encode())
            generated = pipe(text[0])["sample"][0]
            generated.save(os.path.join(validation_dir, f"sample_{i}_image.jpg"))
