import argparse
import clip
import json
import numpy as np
import os
import pandas as pd
import PIL
import sys
import torch

from models import CustomCLIPWrapper, init_img_model, init_txt_model
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path',
    required=True,
    type=str,
    help='Path to a train or test split csv'
)
parser.add_argument(
    '--val_path',
    default=None,
    type=str,
    help='Path to val split csv to append'
)
parser.add_argument(
    '--embed_type',
    required=True,
    type=str,
    help='Choose (image | text)'
)
parser.add_argument(
    '--save_as',
    required=True,
    type=str,
    help='filename for embeddings csv'
)
parser.add_argument(
    '--chexpert_folder',
    default=None,
    type=str,
    help='Path to folder with chexpert labelled sentences as csv files'
)
parser.add_argument(
    '--write_after',
    default=5000,
    type=int,
    help='Output csv in batches of at least --write_after size'
)
parser.add_argument(
    '--config_file',
    required=True,
    type=str,
    help='Path to config json'
)
parser.add_argument(
    '--batch_size',
    default=1,
    type=int,
    help='Batch size'
)
parser.add_argument(
    '--num_workers',
    default=0,
    type=int,
    help='Number of workers'
)


class TextDataset(Dataset):
    def __init__(
        self,
        data: str,
        val_data=None,
        chexpert_data=None,
        split_reports=False,
        shuffle=False,
    ):
        """Create a text dataset from a csv file with reports.
        Args:
            data (str): Path to a csv file with image paths and reports.
            val_data (str, optional): Path to a csv file with validation image paths and reports.
            chexpert_data (str, optional): Path to the folder with the chexpert csv files.
            split_reports (bool, optional): Whether or not reports should be split by sentences.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        self.chexpert_data = chexpert_data
        self.data = pd.read_csv(data, index_col=0)[["study_id", "report"]]

        print("Data len:", len(self.data))
        # append the validation set to the train set
        if val_data is not None:
            val_data = pd.read_csv(val_data, index_col=0)[["study_id", "report"]]
            self.data = pd.concat([self.data, val_data], ignore_index=True)
            print("Len after adding val data:", len(self.data))
        self.data = self.data.dropna().reset_index(drop=True)
        print("After dropping nans:", len(self.data))
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        print("After dropping duplicates:", len(self.data))

        # assuming the chexpert files are stored as a series of csv files
        if self.chexpert_data is not None:
            chexpert_folder = Path(self.chexpert_data)
            chex_files = [*chexpert_folder.glob("**/*.csv")]
            print(f"Found {len(chex_files)} chex csvs")
            dfs = []
            for chex_file in chex_files:
                dfs.append(pd.read_csv(chex_file))
            chex = pd.concat(
                dfs, axis=0, ignore_index=True
            )  # want to explicitly keep indices (the study_ids)
            chex = chex.dropna()

            # every study is joined with every one of the chexpert-classified sentences (1 to M join)
            # dropping the full reports
            self.data = self.data.merge(
                chex, left_on="study_id", right_on="mimic_id"
            ).drop(columns=["index", "report_x", "mimic_id", "report_y", "cat", "vals"])
            self.data = self.data.rename(columns={"sents": "report"})
            self.data = self.data.drop_duplicates().reset_index(drop=True)
            print("Len after splitting by chexpert sentences:", len(self.data))

        if split_reports:
            print("splitting reports by sentence")
            self.data = self.split_reports(self.data)
            print("Len after splitting by sentences", len(self.data))

        self.keys = list(self.data.index)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]

        study_id = self.data.loc[key]["study_id"]

        text = str(self.data.loc[key]["report"])
        text = text.replace("\n", "").replace("\r", "")

        item = {"study_id": study_id, "text": text}

        # Success
        return item

    def split_reports(self, data):
        data["report"] = data["report"].str.split(".")
        data = data.explode("report").reset_index(drop=True)
        data = data.replace("", np.nan).dropna().reset_index(drop=True)
        return data


class CLIPTextDataset(TextDataset):
    def __init__(
        self,
        data: str,
        val_data=None,
        chexpert_data=None,
        split_reports=False,
        shuffle=False,
    ):
        """Create a CLIP specific text dataset from a csv file with reports.
        Args:
            data (str): Path to a csv file with image paths and reports.
            val_data (str, optional): Path to a csv file with validation image paths and reports.
            chexpert_data (str, optional): Path to the folder with the chexpert csv files.
            split_reports (bool, optional): Whether or not reports should be split by sentences.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__(data, val_data, chexpert_data, split_reports, shuffle)

    # main difference is that CLIP requires tokenization at this stage
    def __getitem__(self, ind):
        item = super().__getitem__(ind)
        item["tokens"] = clip.tokenize(item["text"], truncate=True)[0]
        return item


class ImageDataset(Dataset):
    def __init__(
        self,
        data: str,
        val_data=None,
        image_size=224,
        shuffle=False,
    ):
        """Create an image dataset from a csv file with image paths.
        Args:
            data (str): Path to a csv file with image paths and reports.
            val_data (str, optional): Path to a csv file with image paths and reports.
            image_size (int, optional): The size of outputted images. Defaults to 224.
            shuffle (bool, optional): Whether or not to have shuffling behavior during sampling. Defaults to False.
        """
        super().__init__()
        self.shuffle = shuffle
        self.data = pd.read_csv(data, index_col=0)[["study_id", "path"]]
        print("Data len:", len(self.data))

        # append the validation set to the train set
        if val_data is not None:
            val_data = pd.read_csv(val_data, index_col=0)[["study_id", "path"]]
            self.data = pd.concat([self.data, val_data], ignore_index=True)
            print("Len after adding val data:", len(self.data))
        self.data = self.data.dropna().reset_index(drop=True)
        print("After dropping nans:", len(self.data))
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        print("After dropping duplicates:", len(self.data))

        self.keys = list(self.data.index)

        self.image_transform = T.Compose(
            [
                T.Lambda(self.fix_img),
                T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(1.0, 1.0)),
                T.ToTensor(),
                T.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )  # ResNet50 values
            ]
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, ind):
        key = self.keys[ind]

        study_id = self.data.loc[key]["study_id"]
        image_filename = self.data.loc[key]["path"]

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_filename))
        except (PIL.UnidentifiedImageError, OSError):
            print(
                f"An exception occurred trying to load file {image_filename} at index {ind}. Exiting..."
            )
            exit(1)

        # return study_id, image_filename, image_tensor, text
        item = {
            "study_id": study_id,
            "image_filename": image_filename,
            "image_tensor": image_tensor,
        }

        # Success
        return item

    def fix_img(self, img):
        return img.convert("RGB") if img.mode != "RGB" else img


# liberally borrowed from https://github.com/rom1504/clip-retrieval/blob/main/clip_retrieval/clip_inference/mapper.py
class CLIPModel:

    def __init__(self, model_path, image_encoder, text_encoder, tokenizer=None):
        """transforms images and texts into clip embeddings"""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using:", self.device)
        self.model = CustomCLIPWrapper.load_from_checkpoint(
            checkpoint_path=model_path,
            image_encoder=image_encoder,
            text_encoder=text_encoder,
        ).to(self.device)
        self.tokenizer = tokenizer

    def __call__(self, item, embed_type="image"):
        self.model.eval()
        with torch.no_grad():
            study_id = item["study_id"]
            if embed_type == "image":
                target = item["image_filename"]
                image_features = self.model.model.encode_image(
                    item["image_tensor"].to(self.device)
                )
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embs = image_features.cpu().numpy()
            elif embed_type == "text":
                target = item["text"]
                if self.model.using_clip:
                    text_tokens = item["tokens"].to(self.device)
                else:
                    text_tokens = self.tokenizer(
                        target,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",
                    ).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                embs = text_features.cpu().numpy()
            else:
                print("invalid embed_type -- choose image or text, exiting...")
                exit(1)

            return {"study_id": study_id, "target": target, "embs": embs}


def save_to_pickle(outs, save_to):
    # saving them in a df
    study_id = []
    target = []
    embs = []
    for out in outs:
        study_id += out["study_id"]
        target += out["target"]
        embs += list(out["embs"])
    embeddings = pd.DataFrame({"study_id": study_id, "target": target, "embs": embs})
    embeddings.to_pickle(save_to)


def save_embeddings(
    model,
    dataset,
    save_to,
    embed_type="image",
    write_after=50000,
    batch_size=1,
    num_workers=0,
):

    dl = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False
    )
    # computing embeddings
    outs = []
    write_after = write_after
    count = 0
    unwritten = False
    for batch in tqdm(dl):
        unwritten = True
        outs.append(model(batch, embed_type))
        if len(outs) * batch_size >= write_after:
            save_path = Path(os.path.join(save_to, f"{count}.pkl"))
            save_to_pickle(outs, save_path)
            outs = []
            count += 1
            unwritten = False
    if unwritten:
        save_path = Path(os.path.join(save_to, f"{count}.pkl"))
        save_to_pickle(outs, save_path)


def main(args):
    args = parser.parse_args(args)
    embed_type = args.embed_type
    save_as = args.save_as
    data_path = args.data_path
    val_path = args.val_path
    chexpert_folder = args.chexpert_folder
    write_after = args.write_after
    batch_size = args.batch_size
    num_workers = args.num_workers

    with open(args.config_file) as f:
        config = json.load(f)

    for c in config["models"]:

        using_clip = False
        if c["image_encoder"] == "clip" or c["text_encoder"] == "clip":
            using_clip = True
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clp, _ = clip.load("RN50", device=device)

            for p in clp.parameters():
                p.data = p.data.float()
                if p.grad:
                    p.grad.data = p.grad.data.float()

            image_encoder = clp.visual
            text_encoder = clp.transformer
            model = CLIPModel(
                model_path=c["model_path"],
                image_encoder=image_encoder,
                text_encoder=text_encoder,
            )
        else:
            image_encoder, _ = init_img_model(c["image_encoder"], c["embed_dim"])
            text_encoder, tokenizer = init_txt_model(
                c["text_encoder"], c["embed_dim"], add_projection=c["add_projection"]
            )
            model = CLIPModel(
                model_path=c["model_path"],
                image_encoder=image_encoder,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )

        name = c["image_encoder"][:3] + c["text_encoder"][:3]
        embs_dir = Path(f"out_{name}")
        embs_subdir = Path(f"{save_as}_{embed_type}")
        save_to = Path(os.path.join(embs_dir, embs_subdir))
        if not save_to.exists():
            os.makedirs(save_to)

        if embed_type == "image":
            data = ImageDataset(data=data_path, val_data=val_path)
        elif embed_type == "text":
            if using_clip:
                data = CLIPTextDataset(
                    data=data_path, val_data=val_path, chexpert_data=chexpert_folder
                )
            else:
                data = TextDataset(
                    data=data_path, val_data=val_path, chexpert_data=chexpert_folder
                )
        else:
            print("Please choose embed_type = (image | text), exiting...")
            exit(1)
        save_embeddings(
            model,
            data,
            save_to=save_to,
            embed_type=embed_type,
            write_after=write_after,
            batch_size=batch_size,
            num_workers=num_workers,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
