import torchtext
import numpy as np
import linecache
import torch
import h5py
from absl import logging
import gzip
import os
from multiprocessing import Pool, Process, Manager
import random
import time
from tqdm import tqdm


class Block(torch.utils.data.Dataset):
    def __init__(self, block_path, in_memory=True):
        self.block_path = block_path
        self.len = None
        self.in_memory = in_memory

        if self.in_memory:
            self.full_tensor = self.get_full_tensor()
            self.len = len(self.full_tensor)

    def __len__(self):
        if self.len is None:
            with h5py.File(
                self.block_path,
                "r",
            ) as h:
                n = h["block"]
                self.len = len(n)
        return self.len

    def get_full_array(self):
        """
        Returns a tensor containing directly the whole block.
        Garbage-collected by PyTorch if not used, so it shouldn't bloat RAM.
        """
        try:
            with h5py.File(
                self.block_path,
                "r",
            ) as h:
                n = np.copy(h["block"])
        except Exception as e:
            logging.info(f"Error getting block: {e}.\n Block: {self.block_path}")
            return None
        return n

    def get_full_tensor(self):
        """
        Returns a tensor containing directly the whole block.
        Garbage-collected by PyTorch if not used, so it shouldn't bloat RAM.

        Warning: the user id and asin are mapped in a weird way by torch
        """
        if self.in_memory:
            return self.full_tensor
        try:
            with h5py.File(
                self.block_path,
                "r",
            ) as h:
                n = h["block"]
                # t = torch.Tensor(n).to(dtype=torch.long)
                t = torch.Tensor(n)
        except Exception as e:
            logging.info(f"Error getting block: {e}.\n Block: {self.block_path}")
            return None
        # We close the HDF5 file but we keep the tensor we copied
        return t

    def __getitem__(self, index):
        """
        Method called by DataLoader.
        We could also create the tensor once (in RAM) at block init, and reference a slice later,
        however it might not work for bigger datasets where all the blocks can't fit in memory
        together, Instead, we open the h5 file, read the corresponding line and close the file.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        if self.in_memory:
            return self.full_tensor[index]

        try:
            with h5py.File(
                self.block_path,
                "r",
            ) as h:
                n = h["block"]
                # t = torch.Tensor(n[index]).to(dtype=torch.long)
                t = torch.Tensor(n[index])

        except Exception as e:
            logging.info(f"Error getting block: {e}.\n Block: {self.block_path}")
            return None
        # We close the HDF5 file but we keep the tensor we copied
        return t


class PreloadedBlock(Block):
    def __init__(self, tensor):
        self.block_path = ""
        self.len = len(tensor)
        self.in_memory = True
        self.full_tensor = tensor


class BlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        blocks_dir,
        max_blocks_in_memory=50_000,
        from_h5="",
        block_names=None,
    ):
        self.blocks = []
        self.max_blocks_in_memory = max_blocks_in_memory

        if from_h5:
            logging.info(f"Preloading from a single H5 file directly in memory")
            # NOTE: 150k blocks seems to be a maximum for Amazon on 32G (186k but keep margin)

            if block_names is None:
                # If no list of block is provided, we parse the block dir
                block_names = []
                for block_name in os.listdir(blocks_dir):
                    block_names.append(block_name.split(".")[0])

            with h5py.File(from_h5, "r") as j:
                for block_name in tqdm(block_names):
                    try:
                        n = np.copy(j[block_name])
                        b = PreloadedBlock(torch.Tensor(n))
                        if len(b) > 0:
                            self.blocks.append(b)
                        else:
                            raise Exception("Empty block.")
                    except:
                        logging.info(f"Error for block: {block_name}")
        else:
            logging.info(f"Loading blocks from different files.")

            for i, block_path in enumerate(
                tqdm(
                    [
                        os.path.join(blocks_dir, block_name)
                        for block_name in os.listdir(blocks_dir)
                    ]
                )
            ):
                b = Block(block_path, in_memory=(i < self.max_blocks_in_memory))
                if len(b) > 0:
                    self.blocks.append(b)
                else:
                    logging.info(f"Empty block: {block_path}")

        logging.info(f"Initialized {len(self.blocks)} blocks.")


class UserTimeLevelDataset(BlockDataset):
    def __init__(
        self,
        blocks_dir,
        timeframe=0,
        max_blocks_in_memory=50_000,
        from_h5="",
        block_names=None,
        max_user_contribution=-1,
    ):

        super(UserTimeLevelDataset, self).__init__(
            blocks_dir, max_blocks_in_memory, from_h5, block_names
        )

        self.timeframe = timeframe
        self.n_events = None
        self.max_user_contribution = max_user_contribution
        self.users = list(self._get_users_indices().values())

    def _get_block_user_indices(self, id_and_block):
        block_id, block = id_and_block
        users = {}

        # All the reviews in one block have the same timestamp here (can be changed)
        time_bucket = get_time_bucket(review=block[0], timeframe=self.timeframe)
        tensor = block.full_tensor if block.in_memory else block.get_full_tensor()
        for event_index, review in enumerate(tensor):
            user = int(review[1])
            block_and_index = (block_id, event_index)
            if (user, time_bucket) in users:
                # Drop additional points if they make the user contribution too important
                if (
                    self.max_user_contribution == -1
                    or len(users[(user, time_bucket)]) < self.max_user_contribution
                ):
                    users[(user, time_bucket)].append(block_and_index)
            else:
                users[(user, time_bucket)] = [block_and_index]
        return users

    def _get_users_indices(self):
        # Naive solution, might be sufficient. Otherwise, keep index in a preprocessed file.
        users = {}
        t = time.time()
        logging.info("Computing users sequentially.")
        users_per_block = []
        for i, block in enumerate(self.blocks):
            users_per_block.append(self._get_block_user_indices((i, block)))

        # logging.info("Computing users in parallel.")
        # pool = Pool()
        # users_per_block = pool.map(self._get_block_user_indices, enumerate(self.blocks))
        # pool.close()
        # pool.join()

        # print(type(users_per_block))
        # print(users_per_block)
        logging.info("Merging user dicts.")
        users = {}
        for user_dict in users_per_block:
            users.update(user_dict)
        tt = time.time()
        logging.info(f"Users computed in {tt - t}s")
        return users

    def __len__(self):
        "Total number of users."
        return len(self.users)

    def get_n_events(self):
        if self.n_events is None:
            logging.info("Counting the number of events")
            self.n_events = sum([len(block) for block in self.blocks])
            logging.info(f"Done: {self.n_events}")
        return self.n_events

    def __getitem__(self, user_index):
        """
        Returns a random sample belonging to this user.
        Warning: this is not the usual semantic for PyTorch datasets.
        This __get__ method is going to be called by another random
        batch sampler, in charge of selecting a random batch of users.
        """

        (block_id, event_index) = random.choice(self.users[user_index])
        return self.blocks[block_id][event_index]

    def substract_split(self, split_ratio=0.8):
        """
        Modifies the current dataset inplace by substracting an event-level test dataset.
        Warning: non-conventional semantic.
        """

        train_ratio, test_ratio, val_ratio = torchtext.data.dataset.check_split_ratio(
            split_ratio
        )

        # print(f"users: {self.users}")
        blocks_indices_users = [
            (event, user_id)
            for user_id, user in enumerate(self.users)
            for event in user
        ]

        N = len(blocks_indices_users)
        randperm = np.random.permutation(N)
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        def get_event(rand_index):
            return blocks_indices_users[rand_index][0]

        test_blocks_and_indices = list(
            map(
                get_event,
                randperm[train_len : train_len + test_len],
            )
        )
        valid_blocks_and_indices = list(
            map(get_event, randperm[train_len + test_len :])
        )

        test_dataset = SubBlockdataset(
            parent_dataset=self, block_and_indices=test_blocks_and_indices
        )
        valid_dataset = SubBlockdataset(
            parent_dataset=self, block_and_indices=valid_blocks_and_indices
        )

        # We remove the users from the training set, but keep the blocks unchanged
        new_users = {}
        for rand_index in randperm[:train_len]:
            event, user = blocks_indices_users[rand_index]
            if user in new_users:
                new_users[user].append(event)
            else:
                new_users[user] = [event]
        # Overwrite the list of events per user
        self.users = list(new_users.values())

        return test_dataset, valid_dataset


class EventWithBoundedUsersDataset(UserTimeLevelDataset):
    """Like UserTimeLevelDataset excepts that __getitem__ and __len__ iterate over events
    instead of iterating over users with sampling.
    This is useful if you want to bound the user contribution and compute statistics.
    """

    def __init__(
        self,
        blocks_dir,
        timeframe=0,
        max_blocks_in_memory=50_000,
        from_h5="",
        block_names=None,
        max_user_contribution=-1,
    ):
        super().__init__(
            blocks_dir,
            timeframe=timeframe,
            max_blocks_in_memory=max_blocks_in_memory,
            from_h5=from_h5,
            max_user_contribution=max_user_contribution,
            block_names=block_names,
        )

        self.cumulated_len = []
        self.len = 0
        self.user_indices = {}

        logging.info(f"Precomputing indices.")
        self._compute_indices()

        logging.info(f"Dataset initialized.")

    def _compute_indices(self):
        start_index = 0
        for user_number, user_samples in enumerate(self.users):
            l = len(user_samples)
            self.len += len(user_samples)
            self.cumulated_len.append(self.len)
            for sub_index, index in enumerate(range(start_index, start_index + l)):
                self.user_indices[index] = (user_number, sub_index)
            start_index += l
        return

    def __len__(self):
        "Total number of samples"
        return self.len

    def __getitem__(self, index):
        "Generates one sample of data"
        index = index % self.len
        (u, i) = self.user_indices[index]
        (block_id, event_index) = self.users[u][i]
        return self.blocks[block_id][event_index]


class EventLevelDataset(BlockDataset):
    def __init__(
        self,
        blocks_dir,
        max_blocks_in_memory=50_000,
        from_h5="",
        block_names=None,
    ):
        super(EventLevelDataset, self).__init__(
            blocks_dir, max_blocks_in_memory, from_h5, block_names
        )

        self.cumulated_len = []
        self.len = 0
        self.block_indices = {}

        logging.info(f"Precomputing indices.")
        self._compute_indices()

        logging.info(f"Dataset initialized.")

    def _compute_indices(self):
        start_index = 0
        for block_number, b in enumerate(self.blocks):
            l = len(b)
            self.len += len(b)
            self.cumulated_len.append(self.len)
            for sub_index, index in enumerate(range(start_index, start_index + l)):
                self.block_indices[index] = (block_number, sub_index)
            start_index += l
        return

    def __len__(self):
        "Total number of samples"
        return self.len

    def _slow_get_block_index(self, index):
        b = 0
        while self.cumulated_len[b] - 1 < index:
            b += 1
        if b == 0:
            i = index
        else:
            i = index - self.cumulated_len[b - 1]
        return (b, i)

    def _get_block_index(self, index):
        # Binary search (slower than just preprocessing)
        if index < self.cumulated_len[0]:
            return (0, index)

        b_min = 1
        b_max = len(self.blocks)
        b = b_min

        while not (
            self.cumulated_len[b - 1] <= index and index < self.cumulated_len[b]
        ):
            if index < self.cumulated_len[b - 1]:
                b_max = b
            else:
                b_min = b
            b = (b_min + b_max) // 2

        i = index - self.cumulated_len[b - 1]
        return (b, i)

    def __getitem__(self, index):
        "Generates one sample of data"
        index = index % self.len
        # (b, i) = self._get_block_index(index)
        (b, i) = self.block_indices[index]
        return self.blocks[b][i]

    def split(self, split_ratio=0.7):
        """Create train-test(-valid?) splits from the instance's examples.
        Returns subdatasets.
        """

        train_ratio, test_ratio, val_ratio = torchtext.data.dataset.check_split_ratio(
            split_ratio
        )

        N = self.len
        randperm = np.random.permutation(N)
        train_len = int(round(train_ratio * N))

        # Due to possible rounding problems
        if not val_ratio:
            test_len = N - train_len
        else:
            test_len = int(round(test_ratio * N))

        indices = (
            randperm[:train_len],  # Train
            randperm[train_len : train_len + test_len],  # Test
            randperm[train_len + test_len :],  # Validation
        )

        splits = tuple(
            Subdataset(parent_dataset=self, indices=index)
            for index in indices
            if len(index) > 0
        )

        return splits


class Subdataset(torch.utils.data.Dataset):
    def __init__(self, parent_dataset: torch.utils.data.Dataset, indices):
        self.indices = indices
        self.parent_dataset = parent_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.parent_dataset[self.indices[index]]


class SubBlockdataset(torch.utils.data.Dataset):
    def __init__(self, parent_dataset: UserTimeLevelDataset, block_and_indices):
        self.indices = block_and_indices
        self.parent_dataset = parent_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        (block_id, event_index) = self.indices[index]
        return self.parent_dataset.blocks[block_id][event_index]


def select_blocks_by_timeframe(block_names, n, t, stochastic=False):
    """
    Returns at most n blocks, grouped by (n // t) chunks of t contiguous blocks,
    where each chunk is aligned with 0.

    E.g. if blocks are split by day, t is a number of days (not a Unix timestamp).
    """

    selected_blocks = []

    candidate_chunks = {}
    for block in block_names:
        # Current h5 naming scheme: "timeframe-user_slice", e.g. "0-10.h5"
        name = block[: -len(".h5")].split("-")
        timestamp, user_slice = int(name[0]), int(name[1])
        # Chunks not aligned with 0 would break user-time DP.\
        if t != 0:
            time_bucket = timestamp // t
        else:
            time_bucket = 0
        if (time_bucket, user_slice) in candidate_chunks:
            candidate_chunks[(time_bucket, user_slice)].append(block)
        else:
            candidate_chunks[(time_bucket, user_slice)] = [block]

    # Filter the chunks that are incomplete
    valid_chunks = list(
        filter(lambda l: len(l) == t or t == 0, candidate_chunks.values())
    )

    # Stochastic selection or just first ones
    if t != 0:
        n_chunks = min(n // t, len(valid_chunks))
    else:
        n_chunks = len(valid_chunks)
    if stochastic:
        selected_chunks = random.sample(valid_chunks, n_chunks)
    else:
        selected_chunks = valid_chunks[0:n_chunks]

    selected_blocks = [block for chunk in selected_chunks for block in chunk]
    return selected_blocks


def load_block(block_path):
    f = gzip.GzipFile(block_path, "r")
    n = np.load(file=f)
    f.close()
    return n


def load_h5_block(block_path):
    with h5py.File(
        block_path,
        "r",
    ) as h:
        logging.info(f"h5 laoding {block_path}")
        logging.info(h)
        n = h["block"]
        logging.info(n)
        logging.info(n.shape)
    return n


def get_time_bucket(review, timeframe):
    date = review[0]
    if timeframe <= 0:
        return 0
    return int(date) // int(timeframe)


def clamp_vocabulary(
    text,
    vocab_size=10_000,
    unk_token_index=0,
):
    return torch.where(text < vocab_size, text, torch.full_like(text, unk_token_index))


def split_review_batch(
    batch,
    label_feature="category",
    max_text_len=528,
    merge_summary=True,
    include_len=False,
    custom_vocab=True,
    vocab_size=10_000,
    unk_token_index=0,
):
    """
    Concatenates summary and review text
    """
    date = batch[:, 0]
    user = batch[:, 1]
    asin = batch[:, 2]
    category = batch[:, 3]
    rating = batch[:, 4]

    if label_feature == "category":
        label = category
    elif label_feature == "rating":
        label = rating
    elif label_feature == "binary_rating":
        label = (rating > 4).to(torch.float)

    if merge_summary:
        text = batch[:, 7 : 7 + max_text_len]
        if custom_vocab:
            text = clamp_vocabulary(
                text,
                vocab_size,
                unk_token_index,
            )
        if include_len:
            length = batch[:, 5] + batch[:, 6]
            data = (length, text)
        else:
            data = text
    else:
        summary_text = batch[:, 7 : 7 + 16]
        review_text = batch[:, 7 + 16 : 7 + 16 + max_text_len]
        if custom_vocab:
            summary_text = clamp_vocabulary(
                summary_text,
                vocab_size,
                unk_token_index,
            )
            review_text = clamp_vocabulary(
                review_text,
                vocab_size,
                unk_token_index,
            )
        if include_len:

            summary_len = batch[:, 5]
            review_len = batch[:, 6]
            data = (summary_len, review_len, summary_text, review_text)
        else:
            data = (summary_text, review_text)

    return data, label


if __name__ == "__main__":
    blocks = os.listdir("/datasets/amazon/blocks_user_time")
    print(select_blocks_by_timeframe(blocks, 10, 2))
    print(select_blocks_by_timeframe(blocks, 3, 20))
    print(select_blocks_by_timeframe(blocks, 6, 6))
    print(select_blocks_by_timeframe(blocks, 4000, 3000))
    print(select_blocks_by_timeframe(blocks, 10, 2, stochastic=True))
    print(select_blocks_by_timeframe(blocks, 10, 2, stochastic=True))
    print(select_blocks_by_timeframe(blocks, 10, 2, stochastic=True))
