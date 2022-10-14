import io

import torch
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, pandas_df, img_col_name, text_col_name):
        super().__init__()
        self.pandas_df = pandas_df
        self.img_col_name = img_col_name
        self.text_col_name = text_col_name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.pandas_df.iloc[idx][self.img_col_name]))
        text = self.pandas_df.iloc[idx][self.text_col_name]
        return (text, img_transform(img))


class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = load_dataset("lambdalabs/pokemon-blip-captions", split="train")
    
    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, idx):
        img = self.dataset[idx]["image"]
        text = self.dataset[idx]["text"]
        return (text, img_transform(img))
