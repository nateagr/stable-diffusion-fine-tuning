import io
import random
from argparse import Namespace
from enum import Enum
from typing import List, NamedTuple, Tuple

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import prepare_mask_and_masked_image
from transformers import get_scheduler
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


AUTH_TOKEN = "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"


class Precision(Enum):
    FP32 = 1
    FP16 = 2
    AMP = 3


class FineTuningConfig(NamedTuple):
    enable_grad_vae: bool
    enable_grad_text_encoder: bool
    n_epochs: int
    grad_acc:int
    scheduler_type: str
    lr: float
    warmup: int
    betas: Tuple[int, int]
    eps: float
    weight_decay: float
    ema_decay: float


def load_pipeline(model_id: str, device: str) -> torch.nn.Module:
    """Load stable diffusion pipeline"""
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        # revision="fp16", torch_dtype=torch.float16,
        use_auth_token=AUTH_TOKEN  # pylint: disable=no-member
    ).to(device)
    return pipe


def decode_mask(encoded_mask: List[int], image_width: int, image_height: int) -> np.ndarray:
    nb_pixels = image_height * image_width
    mask = np.zeros((nb_pixels,), dtype=bool)
    mask[encoded_mask] = True
    return mask.reshape((image_height, image_width))


class PandasDataset(torch.utils.data.Dataset):
    def __init__(
        self, pandas_df, tokenizer, text_col_name, img_col_name,
        mask_col_name, dataset_size, resolution, random_flip_thrs
    ):
        super().__init__()
        self.pandas_df = pandas_df
        self.img_col_name = img_col_name
        self.mask_col_name = mask_col_name
        self.text_col_name = text_col_name
        self.dataset_size = dataset_size
        self.random_flip_thrs = random_flip_thrs
        self.tokenizer = tokenizer
        self.resizing = transforms.Resize((resolution, resolution))
        self.horizontal_flip = transforms.functional.hflip
        self.image_transformations = transforms.Compose([
            self.resizing,
            transforms.CenterCrop(resolution)
        ])
        self.tensor_transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        item = self.pandas_df.iloc[idx]
        
        pil_image = Image.open(io.BytesIO(item[self.img_col_name]))
        if not pil_image.mode == "RGB":
            pil_image = pil_image.convert("RGB")
        w, h = pil_image.size
        pil_image = self.image_transformations(pil_image)
            
        decoded_mask = decode_mask(item[self.mask_col_name], w, h)
        decoded_mask = np.where(decoded_mask == True, 0, 255)
        pil_mask = self.image_transformations(Image.fromarray(decoded_mask.astype(np.uint8)))
        
        if random.uniform(0, 1) > self.random_flip_thrs:
            pil_image = self.horizontal_flip(pil_image)
            pil_mask = self.horizontal_flip(pil_mask)
        return {
            "text": item[self.text_col_name],
            "tensor_image": self.tensor_transformations(pil_image),
            "pil_image": pil_image,
            "pil_mask": pil_mask
        }
    

def collate_fn(examples):
    texts = [example["text"] for example in examples]
    tensor_images = torch.stack([example["tensor_image"] for example in examples])
    tensor_images = tensor_images.to(memory_format=torch.contiguous_format).float()
    pil_images = [example["pil_image"] for example in examples]
    pil_masks = [example["pil_mask"] for example in examples]
    masks, masked_images = prepare_mask_and_masked_image(pil_images, pil_masks)
    batch = {
        "texts": texts,
        "tensor_images": tensor_images,
        "masks": masks,
        "masked_images": masked_images
    }
    return batch


def configure_gradient(
    pipe,
    gradient_checkpointing: bool,
    enable_grad_vae,
    enable_grad_text_encoder,
    unet_param_filter
):
    pipe.unet.requires_grad_(False)
    if gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    print(f"Gradient checkpointing enabled: {pipe.unet.is_gradient_checkpointing}")
    params_to_optimize = [y for x, y in pipe.unet.named_parameters() if unet_param_filter(x)]
    for param in params_to_optimize:
        param.requires_grad_(True)
        
    pipe.text_encoder.requires_grad_(enable_grad_text_encoder)
    if enable_grad_text_encoder:
        params_to_optimize += list(pipe.text_encoder.parameters())
    pipe.vae.requires_grad_(enable_grad_vae)
    if enable_grad_vae:
        params_to_optimize += list(pipe.vae.parameters())

    return params_to_optimize


def create_exponentiel_moving_average_model(ema_decay, model):
    if ema_decay > 0:
        ema_model = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
    else:
        ema_model = Namespace(
            average_parameters=lambda: open("/tmp/dontusethisplease", "w"),
            update=lambda: None
        )
    return ema_model


def save_checkpoint(pipe, suffix):
    #if ema_decay:
    #    ema_unet.copy_to(unet.parameters())
    pipe.save_pretrained(f"/home/g.racic/sync/fine_tuned_sd_inpaint/{suffix}")
    
    
def compute_loss(batch, pipe, device, dtype, resolution, vae_scale_factor):
    batch_size = len(batch["texts"])
    timesteps_inds = torch.randint(0, len(pipe.scheduler.timesteps), (batch_size,)).long()
    timesteps = torch.LongTensor(pipe.scheduler.timesteps)[timesteps_inds]
    tokens = pipe.tokenizer(
        batch["texts"], padding="max_length", truncation=True, max_length=pipe.tokenizer.model_max_length
    )
    with torch.inference_mode():
        # Convert images to latent space
        image_latents = pipe.vae.encode(
            batch["tensor_images"].type(dtype).to(device)
        ).latent_dist.sample().cpu() * 0.18215
        
        # Sample noise that we'll add to the latents
        noises = torch.randn_like(image_latents)
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = pipe.scheduler.add_noise(image_latents, noises, timesteps) # or timesteps - 1 ?

        # Convert masked images to latent space
        masked_latents = pipe.vae.encode(
            batch["masked_images"].type(dtype).to(device)
        ).latent_dist.sample().cpu() * 0.18215
        
        # resize the mask to latents shape as we concatenate the mask to the latents
        masks = torch.nn.functional.interpolate(
            batch["masks"], size=(resolution // vae_scale_factor, resolution // vae_scale_factor)
        )
        masks = masks.type(dtype).reshape(-1, 1, resolution // vae_scale_factor, resolution // vae_scale_factor)

        # Concatenate the noisy latents, masks and masked latents
        latent_model_input = torch.cat([noisy_latents, masks, masked_latents], dim=1)
        
    # Text encoder hidden states for conditioning
    encoder_hidden_states = pipe.text_encoder(
        torch.LongTensor(tokens.input_ids).to(device),
        attention_mask=torch.tensor(tokens.attention_mask).type(dtype).to(device)
    )[0]
    
    # Noise prediction
    model_pred = pipe.unet(
        latent_model_input.to(device), timesteps.to(device), encoder_hidden_states=encoder_hidden_states
    ).sample
    
    # Loss
    if pipe.scheduler.config.prediction_type == "epsilon":
        target = noises
    elif pipe.scheduler.config.prediction_type == "v_prediction":
        target = pipe.scheduler.get_velocity(image_latents, noises, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {pipe.scheduler.config.prediction_type}")
    return torch.nn.functional.mse_loss(target.to(device), model_pred, reduction="none").mean([1, 2, 3]).mean()


def run(config: FineTuningConfig):
    model_id = "runwayml/stable-diffusion-inpainting"
    device = "cuda:0"

    pipe = load_pipeline(model_id, device)

    # Dataset
    vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    resolution = pipe.unet.config.sample_size * vae_scale_factor
    random_flip_thrs = 0.5
    batch_size = 1

    training_dataset = "hdfs://root/user/g.racic/annotated_segmentation_dataset_filtered_translated_en"
    dataset_pdf = pd.read_parquet(training_dataset)
    validation_ds_ratio = 0.2
    validation_ds_size = int(len(dataset_pdf) * validation_ds_ratio)
    train_pdf = dataset_pdf[validation_ds_size:]
    train_dataset = PandasDataset(
        train_pdf, pipe.tokenizer, "translated_texts", "branding_image_bytes",
        "sparse_mask", len(train_pdf), resolution, random_flip_thrs
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Configure model and training
    pipe.guidance_scale = 1.0
    gradient_checkpointing = True
    unet_param_filter = lambda x: "norm" in x or "bias" in x or "emb" in x or "attn" in x
    enable_grad_vae = config.enable_grad_vae
    enable_grad_text_encoder = config.enable_grad_text_encoder
    n_epochs = config.n_epochs
    train_steps = len(train_dataset) * n_epochs
    grad_acc = config.grad_acc

    scheduler_type = config.scheduler_type
    warmup = config.warmup
    lr = config.lr
    betas = config.betas
    eps = config.eps
    weight_decay = config.weight_decay

    ema_decay = config.ema_decay

    params_to_optimize = configure_gradient(
        pipe=pipe, gradient_checkpointing=gradient_checkpointing, enable_grad_vae=enable_grad_vae,
        enable_grad_text_encoder=enable_grad_text_encoder, unet_param_filter=unet_param_filter
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
    )
    scheduler = get_scheduler(
        scheduler_type, optimizer, warmup, train_steps
    )
    ema_unet = create_exponentiel_moving_average_model(ema_decay, pipe.unet.parameters())

    nb_steps_between_ckpt = len(train_dataset)
    max_grad_norm = 1.0

    losses = []
    global_i = 0
    bar = tqdm(initial=global_i, total=train_steps)
    scaler = torch.cuda.amp.GradScaler()
    precision = Precision.FP32
    dtype = torch.FloatTensor if precision in [Precision.FP32, Precision.AMP] else torch.HalfTensor
    dtype

    while True:
        total_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            pipe.unet.train()
            torch.cuda.empty_cache()

            # Compute loss
            if precision == Precision.AMP:
                with torch.cuda.amp.autocast():
                    loss = compute_loss(
                        batch=batch, pipe=pipe, device=device, dtype=dtype, resolution=resolution,
                        vae_scale_factor=vae_scale_factor
                    )
            else:
                loss = compute_loss(
                    batch=batch, pipe=pipe, device=device, dtype=dtype, resolution=resolution,
                    vae_scale_factor=vae_scale_factor
                )
            loss /= grad_acc
            losses.append(loss.item())
            total_loss += loss.item()
            bar.set_postfix(loss=total_loss / (i + 1))

            # Compute gradients
            if precision == Precision.AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update model and scheduler
            if i % grad_acc == grad_acc - 1:
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_grad_norm)
                if precision == Precision.AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                ema_unet.update()
                bar.update()
                optimizer.zero_grad()

                if global_i % nb_steps_between_ckpt == nb_steps_between_ckpt - 1:
                    save_checkpoint(pipe, global_i)

                global_i += 1

            if global_i >= train_steps:
                break

        if global_i >= train_steps:
            break

