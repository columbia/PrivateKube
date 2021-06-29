# Modified version of the main training script, adapted for this example.
# This part only keeps the evaluation side

import sys, os, shutil
import h5py
import time
import io
import random
import tempfile
from tqdm import tqdm
from absl import app, flags, logging
from ray.util.multiprocessing import Pool
import gcsfs
import numpy as np
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier


import torchtext
import torch

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import opacus

from privatekube.experiments.datasets import (
    EventLevelDataset,
    split_review_batch,
    UserTimeLevelDataset,
    select_blocks_by_timeframe,
)
from privatekube.experiments.utils import (
    build_flags,
    flags_to_dict,
    load_yaml,
    results_to_dict,
    save_yaml,
    save_model,
    binary_accuracy,
    multiclass_accuracy,
    epoch_time,
)
from privatekube.privacy.text import build_public_vocab
from privatekube.privacy.rdp import (
    compute_noise_from_target_epsilon,
    ALPHAS,
    compute_rdp_sgm,
)
from privatekube.privacy.mechanisms import laplace_sanitize

import models


DEFAULT_DATA_PATH = Path("/PrivateKube/evaluation/macrobenchmark/data")

# Define default args
dataset_args = {
    "n_blocks": 200,
    "max_text_len": 140,
    "vocab_size": 10_000,
    "n_blocks_test": 200,
}

input_path_args = {
    "dataset_dir": "",
    "dataset_monofile": "",
    "block_counts": str(DEFAULT_DATA_PATH.joinpath("block_counts.yaml")),
    "emb_path": str(DEFAULT_DATA_PATH.joinpath(".vector_cache")),
    "model_path": "",
}

model_args = {
    "task": "product",
    "model": "bow",
    "embedding_dim": 100,
    "hidden_dim_1": 240,
    "hidden_dim_2": 195,
    "hidden_dim": 100,
    "dropout": 0.25,
}

training_args = {
    "device": "cuda",
    "learning_rate": 0.01,
    "dp": 0,
    "dp_eval": 0,
    "user_level": 0,
    "epsilon": 5.0,
    "delta": 1e-5,
    "n_epochs": 15,
    "batch_size": 64,
    "virtual_batch_multiplier": 2,
    "adaptive_batch_size": 1,
    "noise": -1.0,
    "timeframe_days": 0,
    "learning_rate_scheduler": 1,
    "dynamic_clipping": 0,
    "max_grad_norm": 1.0,
    "per_layer_clipping": 0,
    "n_workers": 6,
    "non_dp_batch_size": 256,
    "epsilon_fraction": 1.0,
    "delta_fraction": 1.0,
}

output_args = {
    "metrics_path": "",
}

build_flags(dataset_args, model_args, training_args, input_path_args, output_args)
FLAGS = flags.FLAGS


np.random.seed(0)


def build_split_dataset():

    block_dir = tempfile.mkdtemp()
    test_block_dir = tempfile.mkdtemp()

    assert FLAGS.dataset_dir == ""
    logging.info("Listing the block names.")
    all_blocks = list(load_yaml(FLAGS.block_counts).keys())

    # logging.info(f"Selecting {FLAGS.n_blocks_test} test blocks (fixed randomness).")
    # test_blocks = np.random.choice(all_blocks, FLAGS.n_blocks_test, replace=False)

    # for tb in test_blocks:
    #     all_blocks.remove(tb)

    # Use every user to the maximum.
    def sort_by_user(block_name):
        if block_name.endswith(".h5"):
            block_name = block_name[: -len(".h5")]
        name = block_name.split("-")
        user_slice = int(name[1])
        return user_slice

    logging.info(
        f"Selecting as few users as possible.\n Pseudorandom and deterministic (hashed user ids)."
    )
    selected_blocks = sorted(all_blocks, key=sort_by_user)[0 : FLAGS.n_blocks]

    assert FLAGS.dataset_dir == ""

    block_dir = None
    block_names = selected_blocks

    # Store for the logs
    FLAGS.dataset_dir = block_dir
    assert FLAGS.dataset_monofile

    from_h5 = FLAGS.dataset_monofile

    if FLAGS.dp and FLAGS.user_level:
        train_data = UserTimeLevelDataset(
            blocks_dir=block_dir,
            timeframe=FLAGS.timeframe_days * 86400,
            from_h5=from_h5,
            block_names=block_names,
        )
    else:
        train_data = EventLevelDataset(
            blocks_dir=block_dir,
            from_h5=from_h5,
            block_names=block_names,
        )

    # We split with a fixed seed to work on the proper subsets even in other containers
    train_data, test_data = train_data.split([0.8, 0.2])
    valid_data = []
    # test_data, valid_data = test_data.split([0.75, 0.25])
    # logging.info(f"Test size: {len(test_data)}\n Valid size: {len(valid_data)}")

    # Values from the preprocessing
    # (max text len doesn't matter here)
    text_field = torchtext.data.Field(
        batch_first=True,
        use_vocab=True,
        init_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        include_lengths=True,
    )
    build_public_vocab(
        text_field,
        max_size=FLAGS.vocab_size - 4,
        vectors=f"glove.6B.{FLAGS.embedding_dim}d",
        unk_init=torch.Tensor.normal_,
        vectors_cache=FLAGS.emb_path,
    )

    return train_data, test_data, valid_data, text_field


def compute_optimal_batch_size(real_batch_size, dataset_len):
    logging.info(
        f"Computing the optimal batch size. Dataset {dataset_len}, real batch {real_batch_size}"
    )
    # Under approximate
    optimal_batch_size = int(np.sqrt(dataset_len))
    if optimal_batch_size <= real_batch_size:
        return optimal_batch_size, 0
    else:
        return (real_batch_size, optimal_batch_size // real_batch_size)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(text_field):
    INPUT_DIM = len(text_field.vocab)
    word_embeddings = text_field.vocab.vectors
    PAD_IDX = text_field.vocab.stoi[text_field.pad_token]
    UNK_IDX = text_field.vocab.stoi[text_field.unk_token]

    if FLAGS.task == "sentiment":
        output_dim = 1
    elif FLAGS.task == "product":
        output_dim = 11

    if FLAGS.model == "lstm":
        model = models.LSTMClassifier(
            batch_size=FLAGS.batch_size,
            output_size=output_dim,
            hidden_size=FLAGS.hidden_dim,
            vocab_size=INPUT_DIM,
            embedding_length=FLAGS.embedding_dim,
            weights=word_embeddings,
            dropout=FLAGS.dropout,
            dp=FLAGS.dp,
        )
    elif FLAGS.model == "bow":
        model = models.NBOW(
            input_dim=word_embeddings.shape[0],
            emb_dim=FLAGS.embedding_dim,
            output_dim=output_dim,
            pad_idx=PAD_IDX,
            word_embeddings=word_embeddings,
        )
    elif FLAGS.model == "feedforward":
        model = models.FeedforwardModel(
            vocab_size=INPUT_DIM,
            embedding_dim=FLAGS.embedding_dim,
            pad_idx=PAD_IDX,
            H_1=FLAGS.hidden_dim_1,
            H_2=FLAGS.hidden_dim_2,
            D_out=output_dim,
            word_embeddings=word_embeddings,
        )
    elif FLAGS.model == "bert":
        # The dataset has been preprocessed with the bert tokenizer, so the indices should be correct
        logging.info(f"Pad and unk index {PAD_IDX, UNK_IDX}")
        model = models.FineTunedBert.build_new(output_dim=output_dim)
        logging.info(
            f"Model {FLAGS.model} has {count_parameters(model)} trainable parameters."
        )
        # Bert has its own pretrained embeddings
        return model

    pretrained_embeddings = text_field.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.data[UNK_IDX] = torch.zeros(FLAGS.embedding_dim)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(FLAGS.embedding_dim)

    logging.info(
        f"Model {FLAGS.model} has {count_parameters(model)} trainable parameters."
    )

    return model


def train(model, iterator, optimizer, criterion, accuracy_fn):

    epoch_loss = 0
    epoch_acc = 0

    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(iterator)):

        # batch = batch.to(FLAGS.device)

        if FLAGS.task == "sentiment":
            data, label = split_review_batch(
                batch,
                label_feature="binary_rating",
                max_text_len=FLAGS.max_text_len,
                include_len=True,
                vocab_size=FLAGS.vocab_size,
                custom_vocab=(FLAGS.model != "bert"),
            )
            text_lengths, text = data
        elif FLAGS.task == "product":
            text, label = split_review_batch(
                batch,
                label_feature="category",
                max_text_len=FLAGS.max_text_len,
                vocab_size=FLAGS.vocab_size,
                custom_vocab=(FLAGS.model != "bert"),
            )

        text = text.to(device=FLAGS.device, dtype=torch.long)
        label = (
            label.to(device=FLAGS.device, dtype=torch.long)
            if FLAGS.task == "product"
            else label.to(device=FLAGS.device, dtype=torch.float)
        )

        if FLAGS.model == "lstm":
            hidden = model.init_hidden(batch_size=len(batch))
            if isinstance(hidden, tuple):
                hidden = (
                    hidden[0].to(FLAGS.device),
                    hidden[1].to(FLAGS.device),
                )
            else:
                hidden = hidden.to(FLAGS.device)
            outputs = model(text, hidden)
        elif FLAGS.model == "bert":
            PAD_IDX = 0
            inputs = {
                "input_ids": text,
                "labels": label,
                "attention_mask": torch.where(
                    text == PAD_IDX, torch.zeros_like(text), torch.ones_like(text)
                ),
            }
            # logging.info(f"Inputs {inputs}")
            # The model outputs loss, logits
            outputs = model(**inputs)[1]
            # logging.info(f"Outputs {outputs}")
        else:
            outputs = model(text)
            # logging.info(f"Outputs {outputs}")
        if FLAGS.task == "sentiment":
            outputs = outputs.squeeze(1)

        loss = criterion(outputs, label)
        acc = accuracy_fn(outputs.detach(), label)

        loss.backward()

        if FLAGS.dp and FLAGS.virtual_batch_multiplier > 1:
            # NOTE: step is not called at every minibatch, so the RDP accountant need to know this

            if (i + 1) % FLAGS.virtual_batch_multiplier == 0 or (i + 1) == len(
                iterator
            ):
                # For the (virtual_batch_multiplier)th batch, call a clip-noise-step
                optimizer.step()
                optimizer.zero_grad()
            else:
                # For the first (virtual_batch_multiplier - 1) batches, just accumulate the gradients
                optimizer.virtual_step()
        else:
            # Regular optimizer step (either non-DP or DP with no virtual step)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()
        # epoch_loss += loss.detach().item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, accuracy_fn):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            # batch = batch.to(FLAGS.device)
            if FLAGS.task == "sentiment":
                data, label = split_review_batch(
                    batch,
                    label_feature="binary_rating",
                    max_text_len=FLAGS.max_text_len,
                    include_len=True,
                    vocab_size=FLAGS.vocab_size,
                    custom_vocab=(FLAGS.model != "bert"),
                )
                text_lengths, text = data
            elif FLAGS.task == "product":
                text, label = split_review_batch(
                    batch,
                    label_feature="category",
                    max_text_len=FLAGS.max_text_len,
                    vocab_size=FLAGS.vocab_size,
                    custom_vocab=(FLAGS.model != "bert"),
                )

            text = text.to(device=FLAGS.device, dtype=torch.long)
            label = (
                label.to(device=FLAGS.device, dtype=torch.long)
                if FLAGS.task == "product"
                else label.to(device=FLAGS.device, dtype=torch.float)
            )

            if FLAGS.model == "lstm":
                hidden = model.init_hidden(batch_size=len(batch))
                if isinstance(hidden, tuple):
                    hidden = (
                        hidden[0].to(FLAGS.device),
                        hidden[1].to(FLAGS.device),
                    )
                else:
                    hidden = hidden.to(FLAGS.device)
                outputs = model(text, hidden)
            elif FLAGS.model == "bert":
                PAD_IDX = 0
                inputs = {
                    "input_ids": text,
                    "labels": label,
                    "attention_mask": torch.where(
                        text == PAD_IDX, torch.zeros_like(text), torch.ones_like(text)
                    ),
                }
                outputs = model(**inputs)[1]
            else:
                outputs = model(text)
            if FLAGS.task == "sentiment":
                outputs = outputs.squeeze(1)

            # print(f"Training. Outputs: {outputs}, labels: {batch.label}")
            loss = criterion(outputs, label)
            acc = accuracy_fn(outputs, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_validate(
    train_data, valid_data, model, optimizer, criterion, accuracy_fn, scheduler
):
    validation_accuracy_epochs = []
    validation_loss_epochs = []
    training_loss_epochs = []
    training_accuracy_epochs = []

    logging.info(f"n workers: {FLAGS.n_workers}")
    train_iterator = torch.utils.data.DataLoader(
        train_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_workers,
        drop_last=True,
    )

    valid_iterator = torch.utils.data.DataLoader(
        valid_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_workers,
        drop_last=False,
    )

    criterion = criterion.to(FLAGS.device)

    best_valid_loss = float("inf")

    for epoch in range(FLAGS.n_epochs):

        start_time = time.time()
        logging.info(f"Starting epoch {epoch + 1}.")
        train_loss, train_acc = train(
            model, train_iterator, optimizer, criterion, accuracy_fn
        )
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, accuracy_fn)
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut2-model.pt")

        logging.info(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        logging.info(
            f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%"
        )
        scheduler.step(train_loss)
        logging.info(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%"
        )

        validation_accuracy_epochs.append(valid_acc)
        validation_loss_epochs.append(valid_loss)
        training_loss_epochs.append(train_loss)
        training_accuracy_epochs.append(train_acc)

    return (
        training_loss_epochs,
        training_accuracy_epochs,
        validation_loss_epochs,
        validation_accuracy_epochs,
    )


def main(argv):
    start_time = time.time()

    # Convert flags for the epsilon = -1 shortcut
    if FLAGS.dp and FLAGS.epsilon < 0 and FLAGS.noise < 0:
        FLAGS.dp = False

    # No multiprocessing for large datasets (save RAM)
    if FLAGS.n_blocks > 50_000:
        logging.info(f"Large dataset, we use a single thread for the loader.")
        FLAGS.n_workers = 0

    # Build the dataset, either event level or user level
    train_data, test_data, valid_data, text_field = build_split_dataset()
    logging.info(
        f"Number of samples for training: {len(train_data)}, validation: {len(valid_data)} and testing: {len(test_data)}"
    )

    # Build and load the model
    model = build_model(text_field).to(FLAGS.device)
    model.load_state_dict(torch.load(FLAGS.model_path))

    if FLAGS.task == "sentiment":
        criterion = nn.BCEWithLogitsLoss().to(FLAGS.device)
        accuracy_fn = binary_accuracy

    elif FLAGS.task == "product":
        criterion = nn.CrossEntropyLoss()
        accuracy_fn = multiclass_accuracy

    # Evaluate the model
    testing_size = len(test_data)
    test_iterator = torch.utils.data.DataLoader(
        test_data,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.n_workers,
        drop_last=False,
    )

    # DP evaluation
    _, accuracy = evaluate(model, test_iterator, criterion, accuracy_fn)

    if FLAGS.dp:
        if FLAGS.user_level:
            raise NotImplementedError("This demo works at event level.")
        dp_accuracy = laplace_sanitize(
            accuracy,
            sensitivity=1 / testing_size,
            epsilon=FLAGS.epsilon * FLAGS.epsilon_fraction,
        )

        metrics = {
            "dp_accuracy": float(dp_accuracy),
        }
    else:
        metrics = {
            "accuracy": float(accuracy),
        }

    if FLAGS.metrics_path != "":
        save_yaml(FLAGS.metrics_path, metrics)
        logging.info(f"Saved metrics: {FLAGS.metrics_path}")

    logging.info(metrics)


if __name__ == "__main__":
    app.run(main)
