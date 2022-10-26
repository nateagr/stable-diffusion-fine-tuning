from typing import List, NamedTuple

from torchvision import transforms
import torch
from torch.utils.data import Dataset


class ClassDataset(NamedTuple):
    instance_images: List[str]
    instance_prompt: str
    class_images: List[str]
    class_prompt: str


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        class_datasets: List[ClassDataset],
        tokenizer,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.class_datasets = class_datasets

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        # FIXME
        return 1_000_000

    def __getitem__(self, index):
        example = {}
        class_id = index % len(self.class_datasets)
        class_dataset = self.class_datasets[class_id]
        
        instance_image_id = (index // len(self.class_datasets)) % len(class_dataset.instance_images)
        instance_image = class_dataset.instance_images[instance_image_id]
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            class_dataset.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        class_image_id = (index // len(self.class_datasets)) % len(class_dataset.class_images)  
        class_image = class_dataset.class_images[class_image_id]
        example["class_images"] = self.image_transforms(class_image)
        example["class_prompt_ids"] = self.tokenizer(
            class_dataset.class_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        
        return example


class PromptDataset(Dataset):
    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def create_dreambooth_dataloader(
    class_datasets: List[ClassDataset],
    train_batch_size: int,
    tokenizer,
    image_resolution: int,
    center_crop: bool,
    with_prior_preservation: bool
):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataset = DreamBoothDataset(
        class_datasets=class_datasets,
        tokenizer=tokenizer,
        size=image_resolution,
        center_crop=center_crop,
    )

    return torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )