from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class Flickr30kDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.image_paths = dataframe["image_path"].astype(str).tolist()
        self.captions = dataframe["caption"].astype(str).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.captions[idx]


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = [str(p) for p in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)
