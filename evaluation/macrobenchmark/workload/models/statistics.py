import sys, os, shutil
import time
import io
from absl import app, flags, logging
import tempfile
import yaml
from pathlib import Path

import torchtext
from torchtext import data, datasets
import torch

import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import opacus
import diffprivlib
from tqdm import tqdm
import numpy as np

from privatekube.experiments.datasets import (
    EventLevelDataset,
    UserTimeLevelDataset,
    EventWithBoundedUsersDataset,
)
from privatekube.experiments.utils import (
    build_flags,
    flags_to_dict,
    results_to_dict,
    save_yaml,
    load_yaml,
    multiclass_accuracy,
)
from privatekube.privacy.mechanisms import (
    add_gaussian_noise,
    add_laplace_noise,
    laplace_sanitize,
    gaussian_sanitize,
)

from privatekube.privacy.rdp import (
    ALPHAS,
    compute_rdp_laplace_from_noise,
)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent.joinpath("data")

# Define default args
dataset_args = {
    "dataset_dir": "",
    "n_blocks": 10,
    "log_path": "",
    "dataset_monofile": str(DEFAULT_DATA_PATH.joinpath("reviews.h5")),
    "block_counts": str(DEFAULT_DATA_PATH.joinpath("block_counts.yaml")),
}

model_args = {
    "model": "avg_rating",
}

training_args = {
    "dp": "user-time",
    "epsilon": 0.5,
    "delta": 0,
    "batch_size": 2048,
}

build_flags(dataset_args, model_args, training_args)
FLAGS = flags.FLAGS

MAX_TOKENS = 145  # Cf preprocessing
MAX_PER_DAY = 20
TOTAL_USER_MAX = 100
MAX_RATING = 5.0
TIMEFRAME_DAYS = 1  # User-time is User-day


def relative_error(target, output):
    error = []
    if isinstance(target, dict):
        target = np.array(list(target.values()))
        output = np.array(list(output.values()))
    if not isinstance(target, list):
        target = np.array([target])
        output = np.array([output])

    norm = np.linalg.norm(target)
    error = np.linalg.norm(target - output)

    if norm == 0:
        return float("inf")
    return float(np.around(error / norm, 5))


def get_max_days(block_names, block_dir):
    """Number of days with non-zero contribution (public)"""
    user_buckets = {}
    if block_names is None:
        block_names = []
        for b in os.listdir(block_dir)[0 : FLAGS.n_blocks]:
            block_names.append(b[: -len(".h5")])
    for b in block_names:
        name = b.split("-")
        user_slice = int(name[1])
        if user_slice in user_buckets:
            user_buckets[user_slice] += 1
        else:
            user_buckets[user_slice] = 1
    n_days = max(user_buckets.values())
    return n_days


def build_dataset():
    logging.info("Collecting the blocks.")
    if FLAGS.dataset_dir != "":
        block_names = None
        from_h5 = ""
        block_dir = tempfile.mkdtemp()
        all_blocks = os.listdir(FLAGS.dataset_dir)
        for b in all_blocks[0 : FLAGS.n_blocks]:
            os.symlink(os.path.join(FLAGS.dataset_dir, b), os.path.join(block_dir, b))
    else:
        block_names = list(load_yaml(FLAGS.block_counts).keys())[0 : FLAGS.n_blocks]
        from_h5 = FLAGS.dataset_monofile
        block_dir = None

    if FLAGS.dp == "event":
        max_events = 1
    elif FLAGS.dp == "user-time":
        max_events = MAX_PER_DAY
    elif FLAGS.dp == "user":
        n_days = get_max_days(block_names, block_dir)
        max_events = min(MAX_PER_DAY * n_days, TOTAL_USER_MAX)

    if FLAGS.dp == "event":
        dataset = EventLevelDataset(
            blocks_dir=block_dir,
            from_h5=from_h5,
            block_names=block_names,
        )
    else:
        if FLAGS.dp == "user-time":
            dataset = EventWithBoundedUsersDataset(
                blocks_dir=block_dir,
                from_h5=from_h5,
                block_names=block_names,
                timeframe=TIMEFRAME_DAYS * 86400,
                max_user_contribution=max_events,
            )
        elif FLAGS.dp == "user":
            dataset = EventWithBoundedUsersDataset(
                blocks_dir=block_dir,
                from_h5=from_h5,
                block_names=block_names,
                timeframe=0,
                max_user_contribution=max_events,
            )

    return dataset, max_events


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


def avg_rating_per_category(loader):
    avg = {}
    counts = {}
    for batch in tqdm(loader):
        for review in batch:
            cat = int(review[3])
            ratio = int(review[4])
            if cat in counts:
                counts[cat] += 1
                avg[cat] += ratio
            else:
                counts[cat] = 1
                avg[cat] = ratio
    for cat, count in counts.items():
        # Not null, otherwise it wouldn't be in counts
        avg[cat] /= count
    return avg, counts


def avg_rating(loader):
    avg = 0
    count = 0
    for batch in tqdm(loader):
        for review in batch:
            # logging.info(review)
            cat = int(review[3])
            ratio = int(review[4])
            count += 1
            avg += ratio

    if count > 0:
        avg /= count
    return avg, count


def count_tokens(loader):
    count = 0
    for batch in tqdm(loader):
        for review in batch:
            count += int(review[5]) + int(review[6])
    return count


def avg_tokens(loader):
    count = 0
    avg = 0
    for batch in tqdm(loader):
        for review in batch:
            count += 1
            avg += int(review[5]) + int(review[6])
    if count > 0:
        avg /= count
    return avg, count


def std_tokens(loader):
    avg, count = avg_tokens(loader)
    std = 0
    for batch in tqdm(loader):
        for review in batch:
            std += np.square(np.int(review[5]) + np.int(review[6]) - avg)
    if count > 0:
        std /= count
        std = float(np.sqrt(std))
    return std, count


def pint(c):
    return int(max(0, np.rint(c)))


def pfloat(c):
    return float(max(0, c))


def run_query():

    dataset, max_events = build_dataset()
    logging.info("Dataset built.")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=6
    )

    logging.info("Dataloader initialized.")

    logging.info(len(loader))
    logging.info(len(dataset))

    rdp_epsilons = None

    if FLAGS.model == "count_reviews":
        c = len(dataset)

        if FLAGS.dp != "non-dp":
            d = pint(laplace_sanitize(c, sensitivity=max_events, epsilon=FLAGS.epsilon))
        else:
            d = c

        r = {FLAGS.model: d, "relative_error": relative_error(c, d)}

    if FLAGS.model == "count_per_category":
        logging.info("Computing real counts.")
        c = count_per_category(loader)

        if FLAGS.dp != "non-dp":
            logging.info("Adding noise.")

            # Disjoint datasets (an event belongs to only one product)
            d = {}
            for product, count in c.items():
                d[product] = pint(
                    laplace_sanitize(
                        count, sensitivity=max_events, epsilon=FLAGS.epsilon
                    )
                )
            # Parallel RDP composition: take the max curve (here, it just corresponds to the max noise)

        else:
            d = c

        r = {FLAGS.model: d, "relative_error": relative_error(c, d)}

    if FLAGS.model == "avg_rating_per_category":
        avg, counts = avg_rating_per_category(loader)

        if FLAGS.dp != "non-dp":
            d = {}
            for cat, count in counts.items():
                # We can't use the real value of the counts to compute the noise
                d[cat] = pfloat(
                    laplace_sanitize(
                        avg[cat],
                        sensitivity=max_events * MAX_RATING,
                        epsilon=FLAGS.epsilon,
                    )
                )

        else:
            d = avg

        r = {FLAGS.model: d, "relative_error": relative_error(avg, d)}

    if FLAGS.model == "avg_rating":
        avg, count = avg_rating(loader)
        # NOTE: we use the true count here. In practice, we should use the DP count given by the block and pay a tiny additional privacy cost.

        logging.info(f"True avg rating: {avg}")
        if FLAGS.dp != "non-dp":
            d = pfloat(
                laplace_sanitize(
                    avg, sensitivity=max_events / float(count), epsilon=FLAGS.epsilon
                )
            )

        else:
            d = avg

        r = {FLAGS.model: d, "relative_error": relative_error(avg, d)}

    if FLAGS.model == "avg_tokens":
        avg, count = avg_tokens(loader)

        if FLAGS.dp != "non-dp":
            d = pfloat(
                laplace_sanitize(
                    avg,
                    sensitivity=MAX_TOKENS * max_events / float(count),
                    epsilon=FLAGS.epsilon,
                )
            )
        else:
            d = avg

        r = {FLAGS.model: d, "relative_error": relative_error(avg, d)}

    if FLAGS.model == "count_tokens":
        count = count_tokens(loader)

        if FLAGS.dp != "non-dp":
            d = pint(
                laplace_sanitize(
                    count, sensitivity=MAX_TOKENS * max_events, epsilon=FLAGS.epsilon
                )
            )
        else:
            d = count

        r = {FLAGS.model: d, "relative_error": relative_error(count, d)}

    if FLAGS.model == "std_tokens":
        std, count = std_tokens(loader)

        if FLAGS.dp != "non-dp":
            d = float(
                np.sqrt(
                    pfloat(
                        laplace_sanitize(
                            np.square(std),
                            sensitivity=np.square(
                                MAX_TOKENS * max_events / float(count)
                            )
                            * (count - 1),
                            epsilon=FLAGS.epsilon,
                        ),
                    )
                )
            )
        else:
            d = std

        r = {FLAGS.model: d, "relative_error": relative_error(std, d)}

    if FLAGS.dp != "non-dp":
        # NOTE: The RDP budget is the same even for functions with sensitivity > 1
        # Proof: see Proposition 6 of the RDP paper, redo the calculation and
        # perform a variable substitution with dx = Δf du
        r.update(
            {
                "alphas": ALPHAS,
                "rdp_epsilons": compute_rdp_laplace_from_noise(1 / FLAGS.epsilon),
            }
        )

    r.update(flags_to_dict(dataset_args, model_args, training_args))
    return r


def main(argv):

    r = run_query()

    try:
        if FLAGS.log_path != "":
            save_yaml(FLAGS.log_path, r)
        else:
            print(yaml.dump(r))
    except AttributeError:
        print(r)


if __name__ == "__main__":
    app.run(main)
