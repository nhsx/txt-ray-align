import argparse
import json
import nltk
import numpy as np
import os
import pandas as pd
import re
import sys

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case_folders",
    default=[],
    nargs="+",
    help="List of paths to case-specific folders holding the embeddings",
)
parser.add_argument(
    "--chexpert_folder",
    default=None,
    type=str,
    help="Folder holding the chexpert labelled sentences in csv files",
)
parser.add_argument(
    "--print_only",
    action="store_true",
    default=False,
    help="Set to print out previously calculated metrics without recomputing",
)
parser.add_argument(
    "--random_state",
    default=0,
    type=int,
    help="Choose random state for sampling"
)
parser.add_argument(
    "--k",
    default=2,
    type=int,
    help="Choose top-k items to retrieve"
)
parser.add_argument(
    "--test_size",
    default=50,
    type=int,
    help="Choose number of test instances"
)
parser.add_argument(
    "--query_type",
    default="image",
    type=str,
    help="Choose from (image | text)"
)
parser.add_argument(
    "--query_folder",
    default="embs_query_image",
    type=str,
    help="Name of query folder within case_folder",
)
parser.add_argument(
    "--bank_type",
    default="text",
    type=str,
    help="Choose from (image | text)"
)
parser.add_argument(
    "--bank_folder",
    default="embs_bank_text",
    type=str,
    help="Name of bank folder within case_folder",
)


def load_embeddings(folder):
    path = Path(folder)
    files = [*path.glob("**/*.pkl")]
    dfs = []
    for file in files:
        dfs.append(pd.read_pickle(file))
    return pd.concat(dfs, axis=0, ignore_index=True)


def add_chex(embs, chex_folder, embed_type="image", add_report=False):

    # get chex frame
    chex_folder = Path(chex_folder)
    chex_files = [*chex_folder.glob("**/*.csv")]
    # print(f"Found {len(chex_files)} chex csv files")

    dfs = []
    for chex_file in chex_files:
        dfs.append(pd.read_csv(chex_file))
    chex = pd.concat(
        dfs, axis=0, ignore_index=True
    )  # want to explicitly keep indices (the study_ids)
    chex = chex.drop_duplicates().reset_index(drop=True)

    # chex = chex.dropna().drop_duplicates().reset_index(drop=True)
    # print("chex sentences:", len(chex))

    # split the cat col into multiple cols and consolidate
    dummies = pd.get_dummies(chex["cat"]).replace(0, np.nan).mul(chex["vals"], 0)
    chex[dummies.columns] = dummies

    # left merge the consolidated labels into the embs frame making sure no embs are dropped
    if embed_type == "image":
        # this preserves only the study level classes
        chex_multi_label = (
            chex.drop(columns=["vals", "index"])
            .groupby(["mimic_id"])
            .max(numeric_only=True)
            .reset_index()
        )
        embs = pd.merge(
            embs, chex_multi_label, left_on="study_id", right_on="mimic_id", how="left"
        ).drop(columns="mimic_id")
    elif embed_type == "text":
        # this preserves sentence-level classes per study
        chex_multi_label = (
            chex.drop(columns=["vals", "index"])
            .groupby(["mimic_id", "sents"])
            .max(numeric_only=True)
            .reset_index()
        )
        embs = pd.merge(
            embs,
            chex_multi_label,
            left_on=["study_id", "target"],
            right_on=["mimic_id", "sents"],
            how="left",
        ).drop(columns=["mimic_id", "sents"])
    else:
        print("Please choose embed_type = (image | text), exiting...")
        exit(1)

    if add_report:
        embs = pd.merge(
            embs,
            chex[["mimic_id", "report"]].drop_duplicates(),
            left_on="study_id",
            right_on="mimic_id",
            how="left",
        ).drop(columns="mimic_id")

    return embs, dummies


def get_sims_in_batches(embs_query, bank_folder, max_k, unique_text=False):

    test_instances = np.stack(embs_query["embs"].values, axis=0)

    bank_folder = Path(bank_folder)
    bank_files = [*bank_folder.glob("**/*.pkl")]

    top_k_sims_per_file = []
    top_k_rows_per_file = []

    # iterate through each of the bank embedding files
    for bank_file in bank_files:
        embs_bank = pd.read_pickle(bank_file)
        if unique_text:
            embs_bank = embs_bank.drop_duplicates(subset="target").reset_index(
                drop=True
            )

        bank_instances = np.stack(embs_bank["embs"].values, axis=0)
        sims = cosine_similarity(test_instances, bank_instances)

        top_k_sims = (
            []
        )  # holds the top k sims for each instance (len(instance) x max_k)
        top_k_rows = []  # holds the top k rows (as df) for each instance

        # iterate through each query instance and find top_k sims within batch
        for i, sim in enumerate(sims):
            top_k = (-sim).argsort()[:max_k]  # list of indices
            top_k_sims.append(list(sim[top_k]))
            top_k_rows.append(embs_bank.loc[top_k].reset_index(drop=True))

        top_k_sims_per_file.append(top_k_sims)
        top_k_rows_per_file.append(top_k_rows)

    # merging similarities instance-wise
    top_k_sims = []
    for line in zip(*top_k_sims_per_file):
        accum = []
        for ln in line:
            accum = accum + ln
        top_k_sims.append(accum)

    # get the actual top_k similarities across all batches
    top_k_idxs = []
    for i, sim in enumerate(top_k_sims):
        sim = np.array(sim)
        top_k = (-sim).argsort()[:max_k]  # list of indices
        top_k_idxs.append(top_k)
        top_k_sims[i] = sim[top_k]

    # merging top rows instance_wise
    top_k_rows = []
    for line in zip(*top_k_rows_per_file):
        top_k_rows.append(pd.concat(line, axis=0, ignore_index=True))

    # get the actual top_k rows across all batches:
    for i, row in enumerate(top_k_rows):
        top_k_rows[i] = row.loc[top_k_idxs[i]].reset_index(drop=True)

    return top_k_sims, top_k_rows


def eval(
    embs_query,
    sims,
    top_rows,
    chex_folder,
    query_type="image",
    bank_type="image",
    add_bleu=False,
    save_preds=None,
    k=2,
):

    # add the chexpert label to the embeddings - also the original report if needed
    add_report = add_bleu or (save_preds is not None and bank_type == "text")
    len_before = len(embs_query)
    embs_query, dummies = add_chex(
        embs_query,
        chex_folder=chex_folder,
        embed_type=query_type,
        add_report=add_report,
    )
    len_after = len(embs_query)
    print("Change", len_before - len_after)
    skipped = 0

    flat_hits = 0
    precisions = []
    recalls = []
    dups = []
    bleus = []

    preds = []

    # iterate through the top_rows for each query instance and take a subset
    for i, rows in enumerate(tqdm(top_rows)):
        top_k = (-np.array(sims[i])).argsort()[:k]
        top_k_rows = rows.loc[top_k].reset_index(drop=True)

        # add chex labels
        top_k_rows, _ = add_chex(
            top_k_rows, chex_folder=chex_folder, embed_type=bank_type
        )
        nan_rows = 0
        for j in range(len(top_k_rows)):
            if (~top_k_rows.loc[j][dummies.columns].isna()).sum() == 0:
                nan_rows += 1
        if nan_rows == len(top_k_rows):
            skipped += 1
            print("test instance skipped -- top k rows contain no non NaN labels")
            continue

        # drop duplicate content
        top_k_rows_dedup = (
            top_k_rows[dummies.columns].drop_duplicates().reset_index(drop=True)
        )
        dups_dropped = len(top_k_rows) - len(top_k_rows_dedup)

        test_instance = embs_query.loc[i][dummies.columns]
        if (~test_instance.isna()).sum() == 0:
            skipped += 1
            print("test instance skipped -- no non NaN labels")
            continue

        if add_bleu:
            ref_report = embs_query.loc[i]["report"].replace("\n", "").replace("\r", "")
            gen_report = ". ".join(top_k_rows["target"])
            ref_report_clean = re.sub("[^0-9a-zA-Z ]+", " ", ref_report)
            gen_report_clean = re.sub("[^0-9a-zA-Z ]+", " ", gen_report)
            bleu = nltk.translate.bleu_score.sentence_bleu(
                [ref_report_clean.split()], gen_report_clean.split(), weights=(0.5, 0.5)
            )
            bleus.append(bleu)

        # iterate through sentences in top_k_rows_dedup and check if there is some overlap in chex labels
        hits_per_row = [test_instance == t for t in top_k_rows[dummies.columns].values]

        num_preds = (~top_k_rows[dummies.columns].isna()).sum().sum()
        hits = pd.concat(hits_per_row, axis=1).any(axis=1)

        # if any of the sentences has any overlap then flat hit
        if hits.sum() > 0:
            flat_hits += 1

        if save_preds is not None:
            query = (
                embs_query.loc[i]["report"].replace("\n", "").replace("\r", "")
                if bank_type == "text"
                else embs_query.loc[i]["target"]
            )

            if hits.sum() == 0:
                hit_classes = None 
            else:
                hit_classes = hits[hits == True].index.tolist()
            
            pred = {
                "study_id": embs_query.loc[i]["study_id"],
                "query": query,
                "hit_classes": hit_classes,
                "out": top_k_rows["target"].tolist(),
            }
            preds.append(pred)

        # count how many of the preds were good
        precisions.append(hits.sum() / num_preds)

        # count how many of the labels were captured
        recalls.append(hits.sum() / (~test_instance.isna()).sum())

        dups.append(dups_dropped)

    if save_preds is not None:
        with open(os.path.join(save_preds, "preds.json"), "w") as f:
            json.dump(preds, f, indent=4)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_dups = sum(dups) / len(dups)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    metrics = {
        "Flat hit": flat_hits / len(embs_query),
        "hits": flat_hits,
        "Precision": avg_precision,
        "Recall": avg_recall,
        "F1": f1,
        "Duplicates": avg_dups,
        "k": k,
        "test_instances": len_after - skipped,
    }
    if add_bleu:
        avg_bleus = sum(bleus) / len(bleus)
        metrics["BLEU"] = avg_bleus

    return metrics


def get_metrics(
    case,
    random_state=0,
    query_type="image",
    query_folder="embs_query_image",
    bank_type="text",
    bank_folder="embs_bank_text",
    chex_folder="data/chex",
    k=2,
    test_size=50,
):
    save_bank = os.path.join(case, bank_folder)
    save_query = os.path.join(case, query_folder)
    save_preds = case
    chex_folder = chex_folder
    test_size = test_size

    embs_query = load_embeddings(save_query)
    embs_query = embs_query.sample(test_size, random_state=random_state).reset_index(
        drop=True
    )

    print("Number of queries:", len(embs_query))

    sim, rows = get_sims_in_batches(embs_query, save_bank, 10, unique_text=True)

    metrics = eval(
        embs_query,
        sim,
        rows,
        chex_folder=chex_folder,
        query_type=query_type,
        bank_type=bank_type,
        save_preds=save_preds,
        k=k,
    )

    metrics["bank"] = save_bank
    metrics["query"] = save_query
    metrics["test_size"] = test_size
    metrics["random_state"] = random_state

    metric_row = pd.DataFrame([metrics])
    save_metrics_to = os.path.join(save_preds, "metrics.csv")
    if Path(save_metrics_to).exists():
        saved_metrics = pd.read_csv(save_metrics_to, index_col=0)
        metric_row = pd.concat([saved_metrics, metric_row], axis=0, ignore_index=True)
    metric_row.to_csv(save_metrics_to)
    return metrics


def main(args):

    args = parser.parse_args(args)
    cases = args.case_folders
    chex_folder = args.chexpert_folder
    print_only = args.print_only
    random_state = args.random_state
    k = args.k
    test_size = args.test_size
    query_type = args.query_type
    query_folder = args.query_folder
    bank_type = args.bank_type
    bank_folder = args.bank_folder

    if not print_only:
        random_state = 0
        for case in cases:
            print("Getting metrics for: ", case)
            _ = get_metrics(
                case,
                random_state=random_state,
                query_type=query_type,
                query_folder=query_folder,
                bank_type=bank_type,
                bank_folder=bank_folder,
                chex_folder=chex_folder,
                k=k,
                test_size=test_size,
            )

    dfs = []
    for c in cases:
        df = pd.read_csv(os.path.join(c, "metrics.csv"), index_col=0)
        dfs.append(df)

    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    with pd.option_context('display.max_colwidth', None, 'display.max_columns', None): 
        print(dfs)


if __name__ == "__main__":
    main(sys.argv[1:])
