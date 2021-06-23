from pathlib import Path
from numpy.core.shape_base import block
import typer
from loguru import logger
import requests
import gzip
import json
import os
import gcsfs
from tqdm import tqdm
import h5py

import numpy as np
import torch
import torchtext
from transformers import AutoTokenizer

from privatekube.privacy.text import build_public_vocab
from privatekube.experiments.utils import save_yaml

app = typer.Typer()

BASE_URL = "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/"
FILES = {
    "Automotive_5.json.gz": "automotive",
    "Books_5.json.gz": "books",
    "CDs_and_Vinyl_5.json.gz": "cds",
    "Clothing_Shoes_and_Jewelry_5.json.gz": "clothing",
    "Toys_and_Games_5.json.gz": "games",
    "Grocery_and_Gourmet_Food_5.json.gz": "groceries",
    "Home_and_Kitchen_5.json.gz": "home",
    "Movies_and_TV_5.json.gz": "movies",
    "Pet_Supplies_5.json.gz": "pets",
    "Sports_and_Outdoors_5.json.gz": "sports",
    "Tools_and_Home_Improvement_5.json.gz": "tools",
}
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.joinpath("data")
N_USER_BUCKETS = 100
MIN_TIMESTAMP = 1356998400
PUBLIC_MINI_REVIEWS_URI = "gs://privatekube-public-artifacts/data/mini/reviews.h5"
PUBLIC_MINI_REVIEWS_CUSTOM_URI = (
    "gs://privatekube-public-artifacts/data/mini/reviews_custom_vocab.h5"
)
PUBLIC_MINI_COUNTS_URI = "gs://privatekube-public-artifacts/data/mini/block_counts.yaml"


@app.command()
def download(
    json_dir: str = typer.Option("", help="Base path to store the raw data (json.gz)")
):
    if json_dir:
        json_dir = Path(json_dir)
    else:
        json_dir = DEFAULT_DATA_PATH.joinpath("json")
        json_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving data to: {json_dir}")

    for basename in FILES.keys():
        logger.info(f"Downloading: {basename}")
        url = BASE_URL + basename
        target = json_dir.joinpath(basename)

        response = requests.get(url, stream=True)
        handle = open(target, "wb")
        for chunk in response.iter_content(chunk_size=512):
            if chunk:  # filter out keep-alive new chunks
                handle.write(chunk)


@app.command()
def preprocess(
    model_name: str = typer.Option(
        "google/bert_uncased_L-4_H-256_A-4", help="HuggingFace model name"
    ),
    json_dir: str = typer.Option("", help="Base path to store the raw data (json.gz)"),
    h5_path: str = typer.Option(
        "", help="Path to store the preprocessed data (as a single h5 file)"
    ),
    counts_path: str = typer.Option(
        "", help="Path to store the category counts (as a yaml file)"
    ),
    custom_vocab: bool = typer.Option(
        False,
        help="Use a custom vocabulary. If set to False, it will use the model tokenizer.",
    ),
    max_lines: int = typer.Option(
        100_000_000, help="Maximum number of lines to read per file."
    ),
    max_text_len: int = typer.Option(127, help="Number of tokens per review text."),
    max_summary_len: int = typer.Option(
        16, help="Number of tokens per review summary."
    ),
    public_vocab_path: str = typer.Option(
        "", help="Path to a custom vocabulary file (.txt)"
    ),
    vocab_size: int = typer.Option(10_000, help="Custom vocabulary size."),
):

    json_dir = Path(json_dir) if json_dir else DEFAULT_DATA_PATH.joinpath("json")
    # h5_path = Path(h5_path) if h5_path else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    h5_path = Path(h5_path) if h5_path else DEFAULT_DATA_PATH.joinpath("h5")
    h5_path.mkdir(parents=True, exist_ok=True)

    counts_path = (
        Path(counts_path)
        if counts_path
        else DEFAULT_DATA_PATH.joinpath("category_counts.yaml")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )

    category_counts = json_to_numpy_lines(
        json_dir,
        h5_path,
        tokenizer,
        custom_vocab=custom_vocab,
        max_lines=max_lines,
        max_text_len=max_text_len,
        max_summary_len=max_summary_len,
        public_vocab_path=public_vocab_path,
        vocab_size=vocab_size,
    )

    save_yaml(counts_path, category_counts)


def save_block_dict_to_h5(block_dict, h5_path):
    logger.info(f"Writing down the blocks to an H5 file\n {h5_path}.")
    with h5py.File(h5_path, "w") as h:
        for block_name, block_lines in tqdm(block_dict.items()):
            try:
                # We need to know the shape of the block in advance to allocate memory
                m = np.stack(block_lines)
                block = h.create_dataset(
                    block_name,
                    m.shape,
                    dtype=np.int32,
                    compression="gzip",
                )
                block[...] = m
            except Exception as e:
                logger.warning(f"Error {e} for {block_name}")


def print_line_handler(npline):
    logger.info(npline)
    return


def csv_line_handler(npline, output_path=DEFAULT_DATA_PATH):
    day = (npline[0] - MIN_TIMESTAMP) // 86400

    with open(os.path.join(output_path, str(day) + ".csv"), "a") as f:
        np.savetxt(f, npline, fmt="%i", newline=", ")
        f.write("\n")


@app.command()
def convert(
    h5_bert_path: str = typer.Option("", help="Input path."),
    h5_custom_path: str = typer.Option("", help="Output path"),
    public_vocab_path: str = typer.Option(
        "", help="Path to a custom vocabulary file (.txt)"
    ),
    vocab_size: int = typer.Option(10_000, help="Custom vocabulary size."),
    model_name: str = typer.Option(
        "google/bert_uncased_L-4_H-256_A-4", help="HuggingFace model name"
    ),
):
    h5_bert_path = (
        Path(h5_bert_path) if h5_bert_path else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    )
    h5_custom_path = (
        Path(h5_custom_path)
        if h5_custom_path
        else DEFAULT_DATA_PATH.joinpath("reviews_custom_vocab.h5")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
    )
    text_field = torchtext.data.Field(
        batch_first=True,
        use_vocab=True,
        # tokenize=cut,
        init_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        # fix_length=max_text_len,
        include_lengths=True,
    )

    build_public_vocab(
        text_field,
        public_vocab_path=public_vocab_path,
        max_size=vocab_size - 4,
        vectors=f"glove.6B.50d",
        unk_init=torch.Tensor.normal_,
        vectors_cache=DEFAULT_DATA_PATH.joinpath(".vector_cache"),
    )

    def update_token(i):
        token = tokenizer.convert_ids_to_tokens(i)
        if i == tokenizer.cls_token_id:
            token = "<bos>"
        elif i == tokenizer.eos_token_id:
            token = "<eos>"
        elif i == tokenizer.pad_token_id:
            token = "<pad>"
        elif i == tokenizer.unk_token_id:
            token = "<unk>"
        return text_field.vocab.stoi[token]

    def update_block_inplace(bert_tensor, n_reserved_colums=7):
        for n in range(bert_tensor.shape[0]):
            for i in range(n_reserved_colums, bert_tensor.shape[1]):
                # print(n, i)
                bert_tensor[n][i] = update_token(int(bert_tensor[n][i]))

    n_reserved_colums = 7
    with h5py.File(h5_bert_path, "r") as bert_h5:
        block_names = set(bert_h5.keys())
        with h5py.File(h5_custom_path, "w") as custom_h5:
            logger.info("Converting from Bert tokenizer to custom vocab.")
            for blockname in tqdm(block_names):
                m = np.array(bert_h5[blockname])
                for n in range(m.shape[0]):
                    for i in range(n_reserved_colums, m.shape[1]):
                        # print(n, i)
                        m[n][i] = update_token(int(m[n][i]))
                b = custom_h5.create_dataset(
                    blockname,
                    m.shape,
                    dtype=np.int32,
                    compression="gzip",
                )
                b[...] = m


def json_to_numpy_lines(
    json_dir,
    h5_path,
    tokenizer,
    custom_vocab=True,
    max_lines=10_000,
    max_text_len=127,
    max_summary_len=16,
    public_vocab_path=None,
    vocab_size=10_000,
):
    def cut(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[: max_text_len - 2]
        return tokens

    def cut_summary(sentence):
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[: max_summary_len - 2]
        return tokens

    if custom_vocab:
        text_field = torchtext.data.Field(
            batch_first=True,
            use_vocab=True,
            tokenize=cut,
            init_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            fix_length=max_text_len,
            include_lengths=True,
        )

        summary_field = torchtext.data.Field(
            batch_first=True,
            use_vocab=True,
            tokenize=cut_summary,
            init_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            fix_length=max_summary_len,
            include_lengths=True,
        )

    else:
        # We use the tokenizer's vocabulary directly
        text_field = torchtext.data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=cut,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=tokenizer.cls_token_id,
            eos_token=tokenizer.eos_token_id,
            pad_token=tokenizer.pad_token_id,
            unk_token=tokenizer.unk_token_id,
            fix_length=max_text_len,
            include_lengths=True,
        )

        summary_field = torchtext.data.Field(
            batch_first=True,
            use_vocab=custom_vocab,
            tokenize=cut_summary,
            preprocessing=tokenizer.convert_tokens_to_ids,
            init_token=tokenizer.cls_token_id,
            eos_token=tokenizer.eos_token_id,
            pad_token=tokenizer.pad_token_id,
            unk_token=tokenizer.unk_token_id,
            fix_length=max_summary_len,
            include_lengths=True,
        )

    rating_field = torchtext.data.LabelField(
        dtype=torch.long, use_vocab=False, preprocessing=lambda l: int(l)
    )

    # category_field = torchtext.data.LabelField()
    user_field = torchtext.data.Field(
        dtype=torch.long,
        use_vocab=False,
        preprocessing=lambda l: abs(hash(l)),
        sequential=False,
    )
    date_field = torchtext.data.Field(
        use_vocab=False,
        dtype=torch.long,
        sequential=False,
    )
    asin_field = torchtext.data.Field(
        dtype=torch.long,
        use_vocab=False,
        preprocessing=lambda l: abs(hash(l)),
        sequential=False,
    )

    fields = {
        "overall": ("rating", rating_field),
        "reviewText": ("text", text_field),
        "reviewerID": ("user", user_field),
        "unixReviewTime": ("date", date_field),
        "asin": ("asin", asin_field),
        "summary": ("summary", summary_field),
    }

    if custom_vocab:
        build_public_vocab(
            text_field,
            public_vocab_path=public_vocab_path,
            max_size=vocab_size - 4,
            vectors=f"glove.6B.50d",
            unk_init=torch.Tensor.normal_,
            vectors_cache=DEFAULT_DATA_PATH.joinpath(".vector_cache"),
        )
        build_public_vocab(
            summary_field,
            public_vocab_path=public_vocab_path,
            max_size=vocab_size - 4,
            vectors=f"glove.6B.50d",
            unk_init=torch.Tensor.normal_,
            vectors_cache=DEFAULT_DATA_PATH.joinpath(".vector_cache"),
        )

    category_counts = {}
    for cat in FILES.values():
        category_counts[cat] = 0

    for category, filename in enumerate(FILES.keys()):
        logger.info(f"Reading {filename}")
        with gzip.open(Path(json_dir).joinpath(filename), mode="rb") as input_file:
            block_dict = {}

            def add_line_to_block_dict(npline):
                """Hash and modulo to have a uniform distribution of user blocks, while being DP."""
                user_hash = npline[1]
                bucket_number = user_hash % N_USER_BUCKETS
                day = (npline[0] - MIN_TIMESTAMP) // 86400
                m = npline.astype(np.int32)
                block_name = f"{day}-{bucket_number}"
                if block_name in block_dict:
                    block_dict[block_name].append(m)
                else:
                    block_dict[block_name] = [m]

            ignored = 0
            positive = 0
            for i, line in enumerate(tqdm(input_file)):
                if i >= max_lines:
                    break
                try:
                    # Some lines are malformed or empty
                    ex = torchtext.data.Example.fromJSON(line, fields)
                    # logger.info(
                    #     f"Example: {ex}. {ex.date}, {ex.date < 1356998400} {int(ex.date) < 1356998400}{ex.text}"
                    # )
                    if len(ex.text) == 0:
                        raise Exception("Empty line.")

                    # Drop examples before Jan 2013
                    if ex.date < MIN_TIMESTAMP:
                        raise Exception("Too old.")

                    review_text, review_len = fields["reviewText"][1].process([ex.text])
                    summary_text, summary_len = fields["summary"][1].process(
                        [ex.summary]
                    )

                    npline = np.array(
                        [
                            ex.date,
                            ex.user,
                            ex.asin,
                            category,
                            ex.rating,
                            summary_len[0].numpy(),
                            review_len[0].numpy(),
                        ]
                    )

                    npline = np.concatenate(
                        [npline, summary_text[0].numpy(), review_text[0].numpy()]
                    )

                    add_line_to_block_dict(npline)

                    category_counts[FILES[filename]] += 1
                except Exception as e:
                    # logger.debug(e)
                    ignored += 1

                if i % 100000 == 0:
                    logger.info(f"Processed {i} examples, ignored {ignored}.")

            logger.info(f"Finished processing {filename}")
            h5_category_file = h5_path.joinpath(filename.split(".")[0] + ".h5")
            save_block_dict_to_h5(block_dict, h5_category_file)
            del block_dict

        logger.info(f"Ignored {ignored} for {filename}.")

    return category_counts


@app.command()
def merge(
    h5_dir: str = typer.Option(
        "", help="Path to a directory of h5 files (one h5 file per category)"
    ),
    h5_path: str = typer.Option(
        "", help="Path to store the preprocessed data (as a single h5 file)"
    ),
    counts_path: str = typer.Option(
        "", help="Path to store the blocks names and sizes (as a single yaml file)"
    ),
):
    h5_dir = Path(h5_dir) if h5_dir else DEFAULT_DATA_PATH.joinpath("h5")
    h5_path = Path(h5_path) if h5_path else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    counts_path = (
        Path(counts_path)
        if counts_path
        else DEFAULT_DATA_PATH.joinpath("block_counts.yaml")
    )

    category_files = []
    category_blocknames = []
    all_blocknames = set({})
    logger.info("Opening the category files and listing the blocks.")
    for category_filename in h5_dir.glob("*.h5"):
        category_files.append(h5py.File(category_filename, "r"))
        block_names = set(category_files[-1].keys())
        category_blocknames.append(block_names)
        all_blocknames.update(block_names)

    block_counts = {}
    logger.info("Merging the blocks into one file.")
    with h5py.File(h5_path, "w") as h:
        for blockname in tqdm(all_blocknames):
            block = []
            for i, f in enumerate(category_files):
                if blockname in category_blocknames[i]:
                    block.append(f[blockname])
                    # logger.debug(block)
            m = np.concatenate(block, axis=0)
            block_counts[blockname] = len(m)
            # logger.debug(m)
            b = h.create_dataset(
                blockname,
                m.shape,
                dtype=np.int32,
                compression="gzip",
            )
            b[...] = m

    logger.info("Closing the category files.")
    for f in category_files:
        f.close()

    logger.info("Saving the block counts.")
    save_yaml(counts_path, block_counts)


@app.command()
def truncate(
    h5_input_path: str = typer.Option(
        "", help="Path to the preprocessed data (as a single h5 file)"
    ),
    h5_output_path: str = typer.Option(
        "", help="Path to store the truncated preprocessed data (as a single h5 file)"
    ),
    n_days: int = typer.Option(50, help="Number of days to keep."),
):
    h5_input_path = (
        Path(h5_input_path)
        if h5_input_path
        else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    )
    h5_output_path = (
        Path(h5_output_path)
        if h5_output_path
        else DEFAULT_DATA_PATH.joinpath("mini_reviews.h5")
    )
    logger.info(f"Extracting the first {n_days} days of the dataset.")
    with h5py.File(h5_input_path, "r") as h:
        with h5py.File(h5_output_path, "w") as out:
            blocks = h.keys()
            for day in tqdm(range(n_days)):
                for user_bucket in range(N_USER_BUCKETS):
                    blockname = f"{day}-{user_bucket}"
                    m = h[blockname]
                    if blockname in blocks:
                        b = out.create_dataset(
                            blockname,
                            m.shape,
                            dtype=np.int32,
                            compression="gzip",
                        )
                        b[...] = m


@app.command()
def chunk(
    h5_input_path: str = typer.Option(
        "", help="Path to the preprocessed data (as a single h5 file)"
    ),
    output_dir: str = typer.Option(
        "", help="Directory to store the h5 blocks (on file per block)"
    ),
    n_days: int = typer.Option(50, help="Number of days to keep."),
):
    h5_input_path = (
        Path(h5_input_path)
        if h5_input_path
        else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    )
    output_dir = (
        Path(output_dir) if output_dir else DEFAULT_DATA_PATH.joinpath("blocks")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting the first {n_days} days of the dataset.")
    with h5py.File(h5_input_path, "r") as h:
        blocks = h.keys()
        for day in tqdm(range(n_days)):
            for user_bucket in range(N_USER_BUCKETS):
                blockname = f"{day}-{user_bucket}"
                m = h[blockname]
                if blockname in blocks:
                    with h5py.File(output_dir.joinpath(f"{blockname}.h5"), "w") as out:
                        b = out.create_dataset(
                            "block",
                            m.shape,
                            dtype=np.int32,
                            compression="gzip",
                        )
                        b[...] = m


@app.command()
def count(
    h5_path: str = typer.Option(
        "", help="Path to the preprocessed data (as a single h5 file)"
    ),
    counts_path: str = typer.Option(
        "", help="Path to store the blocks names and sizes (as a single yaml file)"
    ),
):
    h5_path = (
        Path(h5_path) if h5_path else DEFAULT_DATA_PATH.joinpath("mini_reviews.h5")
    )
    counts_path = (
        Path(counts_path)
        if counts_path
        else DEFAULT_DATA_PATH.joinpath("mini_block_counts.yaml")
    )
    block_counts = {}
    logger.info("Browsing the blocks.")
    with h5py.File(h5_path, "r") as h:
        for blockname in tqdm(h.keys()):
            block_counts[blockname] = len(h[blockname])

    logger.info("Saving the block counts.")
    save_yaml(counts_path, block_counts)


@app.command()
def getmini(
    h5_path: str = typer.Option(
        "", help="Path to store the preprocessed data (as a single h5 file)"
    ),
    h5_path_custom_vocab: str = typer.Option(
        "",
        help="Path to store the preprocessed data with custom vocabulary (as a single h5 file)",
    ),
    counts_path: str = typer.Option(
        "", help="Path to store the blocks names and sizes (as a single yaml file)"
    ),
):
    h5_path = Path(h5_path) if h5_path else DEFAULT_DATA_PATH.joinpath("reviews.h5")
    h5_path_custom_vocab = (
        Path(h5_path_custom_vocab)
        if h5_path_custom_vocab
        else DEFAULT_DATA_PATH.joinpath("reviews_custom_vocab.h5")
    )

    counts_path = (
        Path(counts_path)
        if counts_path
        else DEFAULT_DATA_PATH.joinpath("block_counts.yaml")
    )
    fs = gcsfs.GCSFileSystem(token="anon")
    logger.info("Downloading a mini version of the dataset")
    fs.get(PUBLIC_MINI_REVIEWS_URI, h5_path)
    fs.get(PUBLIC_MINI_REVIEWS_CUSTOM_URI, h5_path_custom_vocab)

    fs.get(PUBLIC_MINI_COUNTS_URI, counts_path)


if __name__ == "__main__":
    app()