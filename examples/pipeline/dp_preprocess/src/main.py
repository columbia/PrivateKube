# Merge a list of h5 blocks and run some statistics


import argparse
import os
import json
from pathlib import Path
import h5py
import numpy as np
import torch

from loguru import logger
from tqdm import tqdm

from privatekube.experiments.datasets import EventLevelDataset
from privatekube.privacy.mechanisms import laplace_sanitize
from privatekube.experiments.utils import save_yaml


def main(args):

    # Local output files
    for output_path in [args.dataset_monofile, args.statistics, args.block_counts]:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

    # Merge all the blocks
    block_counts = {}
    with h5py.File(args.dataset_monofile, "w") as out:
        logger.info(
            "Created an output h5 file. Starting to browse the input h5 files..."
        )
        for h5_input_path in Path(args.input_data).iterdir():
            with h5py.File(h5_input_path, "r") as h:
                blockname = h5_input_path.stem

                # We don't actually use the count in this demo, but it's easier to keep the same structure
                block_counts[blockname] = 0

                m = h["block"]
                b = out.create_dataset(
                    blockname,
                    m.shape,
                    dtype=np.int32,
                    compression="gzip",
                )
                b[...] = m
        logger.info("Finished writing the dataset.")

    # Compute some statistics
    def count_per_category(loader):
        counts = {}
        for batch in tqdm(loader):
            for review in batch:
                cat = int(review[3])
                if cat in counts:
                    counts[cat] += 1
                else:
                    counts[cat] = 1
        return counts

    def pint(c):
        return int(max(0, np.rint(c)))

    dataset = EventLevelDataset(
        blocks_dir=None,
        from_h5=args.dataset_monofile,
        block_names=list(block_counts.keys()),
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=False, num_workers=6
    )

    logger.info("Dataloader initialized.")

    logger.info(len(loader))
    logger.info(len(dataset))

    logger.info("Computing real counts.")
    c = count_per_category(loader)

    logger.info("Adding noise.")

    # Disjoint datasets (an event belongs to only one product)
    d = {}
    for product, count in c.items():
        d[product] = pint(
            laplace_sanitize(
                count, sensitivity=1, epsilon=args.epsilon * args.epsilon_fraction
            )
        )

    statistics = {
        "count_per_category": d,
        "epsilon": args.epsilon * args.epsilon_fraction,
        "delta": 0,
    }

    # Write the rest of the outputs locally too
    with open(args.statistics, "w") as output_json:
        json.dump(statistics, output_json)

    logger.info("Saving the block counts.")
    save_yaml(args.block_counts, block_counts)


if __name__ == "__main__":

    # Documentation for args: see .yaml component
    parser = argparse.ArgumentParser()

    # Input Values
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--epsilon_fraction", type=float, required=True)
    parser.add_argument("--delta_fraction", type=float, required=True)

    # Output Paths (provided by Kubeflow to write artifacts)
    parser.add_argument("--dataset_monofile", type=str, required=True)
    parser.add_argument("--statistics", type=str, required=True)
    parser.add_argument("--block_counts", type=str, required=True)

    args = parser.parse_args()
    main(args)
