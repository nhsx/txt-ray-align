import argparse
import os
import pandas as pd
import PIL
import sys

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_folder", required=True, type=str, help="Path to deepest common folder with images"
)
parser.add_argument(
    "--resize_to", default=256, type=int, help="Resize images to this size"
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="Batch size for dataloader"
)
parser.add_argument(
    "--num_workers", default=0, type=int, help="Number of workers for dataloader"
)


class ImageDataset(Dataset):
    def __init__(self, image_folder, resize_to=256):
        super().__init__()

        print("Finding images...")
        self.image_files = [*Path(image_folder).glob("**/*.jpg")]

        print("Found:", len(self.image_files))
        self.image_transform = T.Compose(
            [
                T.Resize(size=resize_to),
                # T.ToTensor()
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = str(self.image_files[index])
        resized_image = self.image_transform(PIL.Image.open(image_file))
        return image_file, resized_image


def dl_collate_fn(batch):
    return tuple(zip(*batch))


def resize(image_folder, resize_to=256, batch_size=32, num_workers=0):

    old_parent = os.path.split(image_folder)
    new_parent = (old_parent[0], old_parent[1] + "_res")
    res_image_folder = os.path.join(*new_parent)

    dataset = ImageDataset(image_folder, resize_to)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dl_collate_fn,
    )
    for batch in tqdm(dataloader):
        for i in range(len(batch[0])):
            image_file = batch[0][i]
            resized_image = batch[1][i]

            # find the bottom subdirectory of the full res image folder
            # replace that subdirectory with new subdirectory keeping deeper structures constant
            index = Path(image_file).parts.index(old_parent[1])
            new_path = Path(res_image_folder).joinpath(
                *Path(image_file).parts[index + 1 :]
            )

            # print(new_path)
            if not new_path.parent.exists():
                os.makedirs(new_path.parent)

            resized_image.save(new_path, format="JPEG")

    print("Resized images placed in: ", res_image_folder)


def main(args):
    args = parser.parse_args(args)

    image_folder = args.image_folder
    resize_to = args.resize_to
    batch_size = args.batch_size
    num_workers = args.num_workers
    resize(image_folder, resize_to, batch_size, num_workers)


if __name__ == "__main__":
    main(sys.argv[1:])
