# This script randomly samples from the full train-val-test sets to create a bank and query
# This is an optional step to reduce the complexity of the retrieval task
import os
import pandas as pd
import random

from argparse import ArgumentParser

random.seed(42)


def sample(samples, sample_size_ratio):
    sample_size = int(len(samples) * sample_size_ratio)
    sample_idx = range(len(samples))
    sample_idx_selected = random.sample(sample_idx, sample_size)
    samples = samples.loc[sample_idx_selected].reset_index(drop=True)
    return samples


def main(args):

    data_folder = args.data_folder
    bank_ratio = args.bank_ratio
    query_ratio = args.query_ratio

    train = os.path.join(data_folder, "train_res.csv")
    val = os.path.join(data_folder, "val_res.csv")
    test = os.path.join(data_folder, "test_res.csv")
    train = pd.read_csv(train, index_col=0)
    val = pd.read_csv(val, index_col=0)
    test = pd.read_csv(test, index_col=0)

    bank = pd.concat([train, val], axis=0, ignore_index=True)
    query = test

    print("Bank original:", len(bank))
    print("Query original:", len(query))

    bank = sample(bank, bank_ratio)
    query = sample(query, query_ratio)

    print("Bank original:", len(bank))
    print("Query original:", len(query))

    bank_path = os.path.join(data_folder, "bank.csv")
    query_path = os.path.join(data_folder, "query.csv")

    bank.to_csv(bank_path)
    query.to_csv(query_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_folder",
        required=True,
        type=str,
        help="Path to folder with train / val/ test csv splits",
    )
    parser.add_argument(
        "--bank_ratio", required=True, type=float, help="Choose from [0.0, 1.0]"
    )
    parser.add_argument(
        "--query_ratio", required=True, type=float, help="Choose from [0.0, 1.0]"
    )
    args = parser.parse_args()
    main(args)
