import os
import uuid

from PIL import Image


def generate_image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols*w, rows*h))    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def generate_images_from_prompt(pipe, prompt, output_path, n_samples, cols=6):
    generated_images = []
    for i in range(n_samples):
        print(f"Generating image {i} ...")
        image = pipe(prompt)["sample"][0]
        generated_images.append(image)
    agg_image = generate_image_grid(generated_images, (len(generated_images) - 1) // cols + 1, cols)
    agg_image.save(output_path)


def generate_images_from_prompts(pipe, prompts, output_root_dir, n_samples, cols=6):
    os.makedirs(output_root_dir, exist_ok=True)
    for prompt in prompts:
        print(f"Generating prompt {prompt} ...")
        output_dir = os.path.join(output_root_dir, str(uuid.uuid4()))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "prompt"), "wb") as fd:
            fd.write(prompt.encode())
        generate_images_from_prompt(pipe, prompt, os.path.join(output_dir, "images.jpg"), n_samples, cols)
