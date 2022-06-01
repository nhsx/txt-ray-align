from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import random
import os
import pandas as pd

random.seed(42)


def get_sample_df(df, chosen_sample):
    return df[df["study_id"].isin(chosen_sample)].reset_index(drop=True)


def main(args):

    # finding the images
    image_folder = args.image_folder
    path_img = Path(image_folder).resolve()
    if not os.path.exists(path_img):
        print("Path not found, exiting...")
        exit(1)

    format = args.image_format
    image_files = [*path_img.glob(f"**/*.{format}")]
    print(f"Found {len(image_files)} images")
    if args.file_level:
        study_stem = [Path(image_file).stem for image_file in image_files]
    else:
        study_stem = [Path(image_file).parent.stem for image_file in image_files]

    df_images = pd.DataFrame({"study_id": study_stem, "path": image_files})

    if args.reports_folder is None and args.text_folder is None:
        print("No reports or texts given, exiting...")
        exit(1)

    if args.reports_folder is not None and args.text_folder is not None:
        print("Choose either reports or texts, exiting...")
        exit(1)

    # finding the reports csvs
    if args.reports_folder is not None:
        reports_folder = args.reports_folder
        path_txt = Path(reports_folder).resolve()
        if not os.path.exists(path_txt):
            print("Path not found, exiting...")
            exit(1)

        dfs = []
        text_files = [*path_txt.glob("**/*.csv")]
        print(f"Found {len(text_files)} reports csvs")
        num_reports = 0
        for text_file in text_files:
            dfs.append(
                pd.read_csv(text_file, header=None, names=["study_id", "report"])
            )
            num_reports += len(pd.read_csv(text_file))
        print(f"Found {num_reports} reports")
        df_text = pd.concat(
            dfs, axis=0, ignore_index=False
        )  # want to explicitly keep indices (the study_ids)

    # finding .txt files
    if args.text_folder is not None:
        reports_folder = args.text_folder
        path_txt = Path(reports_folder).resolve()
        if not os.path.exists(path_txt):
            print("Path not found, exiting...")
            exit(1)

        txt_files = [*path_txt.glob("**/*.txt")]
        print(f"Found {len(txt_files)} text files")
        study_stem = []
        texts = []
        for txt_file in tqdm(txt_files):
            study_stem.append(Path(txt_file).stem)
            with open(txt_file) as f:
                text = f.read()
                if args.templating:
                    text = f"this is a photo of {text}."
                texts.append(text)

        df_text = pd.DataFrame({"study_id": study_stem, "report": texts})

    # merging dfs holding image paths and reports
    df = df_images.merge(df_text, left_on="study_id", right_on="study_id")
    print(f"Found {len(df)} matches between images and reports")

    # Take subset based on studies rather than images
    unique_study_ids = pd.unique(df["study_id"].values)
    total_len = len(unique_study_ids)
    subset_fraction = args.subset_fraction
    subset_len = int(total_len * subset_fraction)

    sample = random.sample(list(unique_study_ids), subset_len)
    assert len(sample) > 0

    train_fraction = args.train_fraction
    train_len = int(subset_len * train_fraction)
    # test set as fraction of (1 - train_fraction)
    test_fraction = args.test_fraction
    test_len = int((subset_len - train_len) * test_fraction)

    sample_train = random.sample(sample, train_len)
    assert len(sample_train) > 0
    sample_remainder = list(set(sample) - set(sample_train))
    sample_test = random.sample(sample_remainder, test_len)
    sample_val = list(set(sample_remainder) - set(sample_test))

    print("Train studies:", len(sample_train))
    print("Val studies:", len(sample_val))
    print("Test studies:", len(sample_test))

    assert (
        len(set(sample_train).intersection(sample_test).intersection(sample_val)) == 0
    )

    df_train = get_sample_df(df, sample_train)
    df_val = get_sample_df(df, sample_val)
    df_test = get_sample_df(df, sample_test)

    print("Train images:", len(df_train))
    print("Val images:", len(df_val))
    print("Test images:", len(df_test))

    if args.output_folder is None:
        sample_folder = f"./len{subset_len}_train{train_fraction}_test{test_fraction}"
    else:
        sample_folder = args.output_folder
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    df_train.to_csv(f"{sample_folder}/train.csv")
    df_val.to_csv(f"{sample_folder}/val.csv")
    df_test.to_csv(f"{sample_folder}/test.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to the folder with images in subfolders",
    )
    parser.add_argument(
        "--image_format",
        default="jpg",
        type=str,
        help="Extension of the image format, e.g. jpg or png",
    )

    parser.add_argument(
        "--reports_folder",
        default=None,
        type=str,
        help="Path to the folder with sectioned reports as csv files",
    )
    parser.add_argument(
        "--text_folder",
        default=None,
        type=str,
        help="Path to the folder with raw reports as txt files",
    )
    parser.add_argument(
        "--file_level",
        action="store_true",
        default=False,
        help="Set flag if pairs are on the file level (rather than study level)",
    )
    parser.add_argument(
        "--templating",
        action="store_true",
        default=False,
        help="Set flag for simple templating - this is a photo of x",
    )

    parser.add_argument(
        "--subset_fraction",
        default=1.0,
        type=float,
        help="Fraction of data to allocate for training",
    )
    parser.add_argument(
        "--train_fraction",
        required=True,
        type=float,
        help="Fraction of data to allocate for training",
    )
    parser.add_argument(
        "--test_fraction",
        default=0.0,
        type=float,
        help="Fraction of remaining data to allocate for testing",
    )

    parser.add_argument(
        "--output_folder",
        default=None,
        type=str,
        help="Path to an output folder to override default format ./len<subset_len>_train<train_fraction>_test<test_fraction>",
    )
    args = parser.parse_args()

    main(args)
