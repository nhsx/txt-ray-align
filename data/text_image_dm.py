# Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, sample

import PIL
import clip
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule


class GenericTextImageDataset(Dataset):
    def __init__(self):
        """Abstract class with some common utilities for Text-Image Datasets"""
        super().__init__()

    def fix_img(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


class MIMIC(GenericTextImageDataset):
    def __init__(
        self,
        data: str,
        image_size=224,
        resize_ratio=0.75,
        shuffle=False,
        custom_tokenizer=False,
        num_sentences=1,
    ):
        """Create a text image dataset from a csv file with image paths and reports.
        Args:
            data (str): Path to a csv file with image paths and reports.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (bool, optional): Whether or not there is a custom tokenizer. Defaults to False.
            num_sentences (int): Number of sentences to sample from the report.
        """
        super().__init__()
        self.shuffle = shuffle
        self.num_sentences = num_sentences

        data = Path(data).resolve()
        print(f"Reading in directly from: {data}")
        data = pd.read_csv(data, index_col=0)

        data = data.dropna()
        self.data = data[data["report"].str.contains("[A-Za-z]")].reset_index(
            drop=True
        )  # dropping those with no text

        self.keys = list(self.data.index)

        self.resize_ratio = resize_ratio
        self.image_transform = T.Compose(
            [
                T.Lambda(self.fix_img),
                T.RandomResizedCrop(
                    image_size, scale=(self.resize_ratio, 1.0), ratio=(1.0, 1.0)
                ),
                T.ToTensor(),
                T.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )  # ResNet50 values
                # T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.custom_tokenizer = custom_tokenizer

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]

        image_file = self.data.loc[key]["path"]

        descriptions = str(self.data.loc[key]["report"])
        descriptions = descriptions.replace("\n", "").replace("\r", "")

        # list of sentences split on commas and dots
        descriptions = descriptions.split(".")
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            # description = descriptions # taking entire reports - careful about mem usage
            # description = descriptions[0] # choose first sentence
            description = ". ".join(
                [
                    s.strip()
                    for s in sample(
                        descriptions, min(len(descriptions), self.num_sentences)
                    )
                ]
            )  # randomly choose num_sentences
            # description = choice(descriptions).strip() # randomly choose one of the sentences
        except IndexError:
            print("An exception occurred trying to choose from description.")
            print(f"Skipping index {ind} - study {key}")
            return self.skip_sample(ind)

        tokenized_text = (
            description
            if self.custom_tokenizer
            else clip.tokenize(description, truncate=True)[0]
        )

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError):
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return image_tensor, tokenized_text


class MIMICDataModule(LightningDataModule):
    def __init__(
        self,
        train: str = None,
        val: str = None,
        batch_size: int = 32,
        num_workers=0,
        image_size=224,
        resize_ratio=0.75,
        shuffle=False,
        custom_tokenizer=None,
        num_sentences=1,
    ):
        """Create a text image datamodule from directories with congruent text and image names.

        Args:
            train (str, optional): Path to train csv.
            val (str, optional): Path to a val csv.
            batch_size (int, optional): The batch size of each dataloader.
            num_workers (int, optional): The number of workers in the DataLoader. Defaults to 0.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            resize_ratio (float, optional): Minimum percentage of image contained by resize. Defaults to 0.75.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
            custom_tokenizer (transformers.AutoTokenizer, optional): The tokenizer to use on the text. Defaults to None.
            num_sentences (int): Number of sentences to sample from the report.
        """
        super().__init__()

        if train is None:
            print("No training images given, exiting...")
            exit(1)
        if val is None:
            print("No validation images given, exiting...")
            exit(1)

        self.batch_size = batch_size
        self.train = train
        self.val = val
        self.num_workers = num_workers
        self.image_size = image_size
        self.resize_ratio = resize_ratio
        self.shuffle = shuffle
        self.custom_tokenizer = custom_tokenizer
        self.num_sentences = num_sentences

    def setup(self, stage=None):
        self.dataset = MIMIC(
            data=self.train,
            image_size=self.image_size,
            resize_ratio=self.resize_ratio,
            shuffle=self.shuffle,
            custom_tokenizer=self.custom_tokenizer is not None,
            num_sentences=self.num_sentences,
        )
        self.val_dataset = MIMIC(
            data=self.val,
            image_size=self.image_size,
            resize_ratio=self.resize_ratio,
            shuffle=self.shuffle,
            custom_tokenizer=self.custom_tokenizer is not None,
            num_sentences=self.num_sentences,
        )
        print("Train data: ", len(self.dataset), " Val data: ", len(self.val_dataset))

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.dl_collate_fn,
        )

    def dl_collate_fn(self, batch):
        if self.custom_tokenizer is None:
            return torch.stack([row[0] for row in batch]), torch.stack(
                [row[1] for row in batch]
            )
        else:
            return torch.stack([row[0] for row in batch]), self.custom_tokenizer(
                [row[1] for row in batch],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
