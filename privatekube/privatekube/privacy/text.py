from collections import Counter, OrderedDict
from absl import logging
import os


def build_public_vocab(field, max_size, public_vocab_path=None, *args, **kwargs):
    """
    Public vocabulary for DP preprocessing.
    """

    if public_vocab_path is None or public_vocab_path == "":
        public_vocab_path = os.path.join(
            os.path.dirname(__file__), "google-10000-english.txt"
        )

    counter = Counter()
    logging.info("Reading public vocabulary.")
    with open(public_vocab_path, "r") as f:
        words = f.read().splitlines()

    if len(words) < max_size:
        logging.warn(
            f"The max_size you asked ({max_size}) is too big for list of public words.\n We now set max_size={len(words)}."
        )
        max_size = len(words)

    logging.info("Browsing words.")
    for i in range(max_size):
        # Most frequent words first
        counter[words[i]] = max_size - i + 1

        # We ignore non sequential fields.

        # if not field.sequential:
        #     x = [x]
        # try:
        #     counter.update(x)
        # except TypeError:
        #     counter.update(chain.from_iterable(x))

    logging.info("Building specials and calling Vocab constructor.")
    specials = list(
        OrderedDict.fromkeys(
            tok
            for tok in [
                field.unk_token,
                field.pad_token,
                field.init_token,
                field.eos_token,
            ]
            + kwargs.pop("specials", [])
            if tok is not None
        )
    )

    field.vocab = field.vocab_cls(
        counter, max_size=max_size, specials=specials, **kwargs
    )
