import os
import io
from typing import List
from tempfile import TemporaryDirectory

import fsspec
import torch
from PIL import Image
import numpy as np
import pandas as pd
from diffusers import StableDiffusionInpaintPipeline


def _transform_to_square_image(
    image_rgb: Image.Image,
    resolution: int
) -> np.ndarray:
    w, h = image_rgb.size
    max_dim = max(w, h)
    if max_dim > resolution:
        resize_ratio = max_dim / resolution
        w, h = int(w // resize_ratio), int(h // resize_ratio)
        image_rgb = image_rgb.resize((w, h))
    new_image = Image.new(mode="RGB", size=(resolution, resolution), color=(255, 255, 255))
    new_image.paste(image_rgb, ((resolution - w) // 2, (resolution - h) // 2))
    return new_image


def decode_mask(encoded_mask: List[int], image_width: int, image_height: int) -> np.ndarray:
    nb_pixels = image_height * image_width
    mask = np.zeros((nb_pixels,), dtype=bool)
    mask[encoded_mask] = True
    return mask.reshape((image_height, image_width))


def _inpaint(
    sd_model,
    init_image_pil,
    mask_pil,
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    sd_width: int,
    sd_height: int,
    nb_images: int,
    seed: int,
    device: str
):
    """Inpaint with stable diffusion"""
    init_image_pil = init_image_pil.convert("RGB").resize((sd_width, sd_height))
    mask_pil = mask_pil.convert("RGB").resize((sd_width, sd_height))
    outputs = []
    for i in range(nb_images):
        output = sd_model(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            image=init_image_pil,
            mask_image=mask_pil,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=sd_width,
            height=sd_height,
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(seed + i)
        )
        outputs.append(output.images[0])
    return outputs


def decode_image_and_mask(branding_image_bytes, sparse_mask):
    branding_image_pil = Image.open(io.BytesIO(branding_image_bytes))
    w, h = branding_image_pil.size
    decoded_mask = decode_mask(sparse_mask, w, h)
    decoded_mask = np.where(decoded_mask == True, 0, 255)
    mask_pil = Image.fromarray(decoded_mask.astype(np.uint8))
    return branding_image_pil, mask_pil


def generate_images(
    pipe,
    dataset,
    nb_images_per_row,
    num_inference_steps,
    guidance_scale,
    resolution,
    seed,
    device
):
    outputs = dict()
    for branding_image_bytes, sparse_mask, translated_texts, partner_id, product_id in \
        zip(*[dataset[col] for col in [
        "branding_image_bytes", "sparse_mask", "translated_texts", "partner_id", "product_id"
    ]]):
        branding_image_pil, mask_pil = decode_image_and_mask(branding_image_bytes, sparse_mask)
        branding_image_pil = _transform_to_square_image(branding_image_pil, resolution)
        mask_pil = _transform_to_square_image(mask_pil, resolution)
        with torch.no_grad():
            outputs[(partner_id, product_id)] = _inpaint(
                sd_model=pipe, init_image_pil=branding_image_pil, mask_pil=mask_pil, prompt=translated_texts,
                negative_prompt="", num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                sd_width=resolution, sd_height=resolution, nb_images=nb_images_per_row, seed=seed, device=device
            )
    return outputs


def generate_images_and_store_by_uc(
    pipe,
    dataset,
    nb_images_per_row,
    num_inference_steps,
    guidance_scale,
    resolution,
    seed,
    device,
    output_root_dir
):
    fs, _ = fsspec.core.url_to_fs(output_root_dir, use_listings_cache=False)
    if fs.exists(output_root_dir):
        raise ValueError(f"Output root directory already exists: {output_root_dir}")
    uc_categories = set(dataset["uc_category"])
    for uc_category in uc_categories:
        output_dir = os.path.join(output_root_dir, f"uc={uc_category}")
        sub_dataset = dataset[dataset["uc_category"] == uc_category]
        generated_images = generate_images(
            pipe, sub_dataset, nb_images_per_row, num_inference_steps, guidance_scale, resolution, seed, device
        )
        for (partner_id, product_id), images in generated_images.items():
            product_dir = os.path.join(output_dir, f"{partner_id}_{product_id}")
            fs.mkdir(product_dir, recursive=True)
            products_by_partner = sub_dataset[sub_dataset["partner_id"] == partner_id]
            product = products_by_partner[products_by_partner["product_id"] == product_id].iloc[0]
            with fs.open(os.path.join(product_dir, "prompt"), "w") as fd:
                fd.write(product["translated_texts"])
            branding_image_pil, mask_pil = decode_image_and_mask(
                product["branding_image_bytes"], product["sparse_mask"]
            )
            with fs.open(os.path.join(product_dir, "branding_image.jpg"), "wb") as fd:
                branding_image_pil.save(fd, format="jpeg")
            with fs.open(os.path.join(product_dir, "mask.jpg"), "wb") as fd:
                mask_pil.save(fd, format="jpeg")
            for i, image in enumerate(images):
                with fs.open(os.path.join(product_dir, f"generated_{i}.jpg"), "wb") as fd:
                    image.save(fd, format="jpeg")


def load_pipeline(hdfs_path: str, device: str) -> torch.nn.Module:
    """Load stable diffusion pipeline"""
    with TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, os.path.basename(hdfs_path))
        fs, _ = fsspec.core.url_to_fs(hdfs_path, use_listings_cache=False)
        fs.get(hdfs_path, local_path, recursive=True)
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            local_path
        ).to(device)
        return pipe
                    
                    
def run_inference_on_test_set(
    pipe_hdfs_path: str,
    dataset_hdfs_path: str,
    test_set_ratio: int,
    nb_images_per_row,
    num_inference_steps,
    guidance_scale,
    resolution,
    seed,
    device,
    output_root_dir
):
    print("Downloading and loading pipeline from HDFS ...")
    pipe = load_pipeline(pipe_hdfs_path, device)
    print("Pipeline loaded!")
    print("Loading dataset from HDFS ...")
    dataset_pdf = pd.read_parquet(dataset_hdfs_path)
    print("Dataset loaded ...")
    dataset_pdf["partner_id"] = np.random.randint(0, 1_000_000, size=len(dataset_pdf))
    dataset_pdf["product_id"] = np.random.randint(0, 1_000_000, size=len(dataset_pdf))
    test_ds_size = int(len(dataset_pdf) * test_set_ratio)
    test_pdf = dataset_pdf[:test_ds_size]
    generate_images_and_store_by_uc(
        pipe,
        test_pdf[:4],
        nb_images_per_row,
        num_inference_steps,
        guidance_scale,
        resolution,
        seed,
        device,
        output_root_dir
    )
