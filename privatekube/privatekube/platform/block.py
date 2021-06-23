from time import sleep
from typing import List
from loguru import logger
import pandas as pd

from privatekube.platform.stoppable_thread import StoppableThread, get_stop_flag
from privatekube.platform.privacy_resource_client import PrivacyResourceClient
from privatekube.platform.timer import Timer
from privatekube.platform.privacy_budget import (
    PrivacyBudget,
    RenyiBudget,
    EpsDelBudget,
    ALPHAS,
)

DATA_BLOCK_GROUP = "columbia.github.com"
DATA_BLOCK_VERSION = "v1"
DATA_BLOCK_PLURAL = "privatedatablocks"


class DataBlockMetadata:
    def __init__(
        self,
        budget: PrivacyBudget,
        dataset: str,
        data_source: str,
        start_time: int,
        end_time: int,
        release_period=0,
    ):
        self.budget = budget
        self.dataset = dataset
        self.data_source = data_source
        self.start_time = start_time
        self.end_time = end_time
        self.release_period = release_period

    def name(self):
        return f"{self.dataset}-{self.start_time}-{self.end_time}"


class DatasetMetadata:
    def __init__(self, dataset, data_blocks: List[DataBlockMetadata]):
        self.dataset = dataset
        self.data_blocks = data_blocks


def make_time_partitioned_data_block(
    name,
    namespace,
    dataset,
    start_time,
    end_time,
    data_source,
    initial_budget,
    release_duration=None,
):

    spec = {
        "dataset": dataset,
        "dimensions": [
            {"attribute": "startTime", "numericValue": str(start_time)},
            {"attribute": "endTime", "numericValue": str(end_time)},
        ],
        "dataSource": data_source,
        "initialBudget": initial_budget
        if isinstance(initial_budget, dict)
        else initial_budget.to_dict(),
        "budgetReturnPolicy": "toPending",
    }
    if release_duration is None:
        spec["releaseMode"] = "base"
    else:
        spec["releaseMode"] = "flow"
        spec["flowReleasingOption"] = {
            "duration": release_duration,
        }
    return {
        "apiVersion": DATA_BLOCK_GROUP + "/" + DATA_BLOCK_VERSION,
        "kind": "PrivateDataBlock",
        "metadata": {"name": name, "namespace": namespace},
        "spec": spec,
    }


def convert_blocks_to_df(blocks):
    rows = []
    renyi = None
    for block in blocks:

        start_time = block["spec"]["dimensions"][0]["numericValue"]
        end_time = block["spec"]["dimensions"][1]["numericValue"]
        initial = PrivacyBudget(block["spec"]["initialBudget"])
        if "status" in block:
            available = PrivacyBudget(block["status"]["availableBudget"])
            pending = PrivacyBudget(block["status"]["pendingBudget"])
        else:
            available = PrivacyBudget(block["spec"]["initialBudget"]).empty_copy()
            pending = PrivacyBudget(block["spec"]["initialBudget"])

        if pending.is_renyi() and renyi is None:
            renyi = True
        elif not pending.is_renyi() and renyi is None:
            renyi = False
        elif not renyi is None and pending.is_renyi() != renyi:
            raise Exception("The blocks have inconsistent budget types.")

        r = [start_time, end_time]
        r.extend(initial.to_list())
        r.extend(available.to_list())
        r.extend(pending.to_list())
        rows.append(r.copy())

    columns = ["start time", "end time"]
    if renyi:
        c = []
        for alpha in ALPHAS:
            c.append(f"initial alpha-{alpha}")
        for alpha in ALPHAS:
            c.append(f"available alpha-{alpha}")
        for alpha in ALPHAS:
            c.append(f"pending alpha-{alpha}")
        columns.extend(c)
    else:
        columns.extend(
            [
                "initial eps",
                "initial delta",
                "available eps",
                "available delta",
                "pending eps",
                "pending delta",
            ]
        )

    rows.sort(key=lambda x: x[0])
    return pd.DataFrame(rows, columns=columns)


class BlockGenerator:
    def __init__(self, client: PrivacyResourceClient, timer: Timer):
        self.client = client
        self.timer = timer

    def create_data_blocks(self, dataset_metadata: DatasetMetadata):
        for data_block in dataset_metadata.data_blocks:
            block_name = data_block.name()
            namespace = "privacy-example"
            block = make_time_partitioned_data_block(
                block_name,
                namespace,
                data_block.dataset,
                data_block.start_time,
                data_block.end_time,
                data_block.data_source,
                EpsDelBudget(1, 0.001),
                release_duration=self.timer.duration_in_k8s(data_block.release_period),
            )

            self.client.create_data_block(namespace, block)
            logger.info(f"Data block {namespace}/{block_name} has been created.")

    def create_data_blocks_periodically(
        self, dataset_metadata: DatasetMetadata, namespace: str
    ):
        thread = StoppableThread(
            target=self._create_data_blocks_periodically,
            args=(dataset_metadata, namespace),
        )
        thread.start()
        return thread

    def _create_data_blocks_periodically(
        self, dataset_metadata: DatasetMetadata, namespace: str
    ):
        blocks = []
        for data_block in dataset_metadata.data_blocks:
            block_name = data_block.name()
            namespace = namespace
            block = make_time_partitioned_data_block(
                block_name,
                namespace,
                dataset_metadata.dataset,
                data_block.start_time,
                data_block.end_time,
                data_block.data_source,
                data_block.budget,
                release_duration=self.timer.duration_in_k8s(data_block.release_period),
            )
            blocks.append((data_block.start_time, block))
        blocks.sort(key=lambda x: x[0])
        idx = 0

        stop_flag = get_stop_flag()
        while not stop_flag() and idx < len(blocks):
            start_time, block = blocks[idx]
            now = self.timer.current()
            if start_time <= now:
                self.client.create_data_block(block["metadata"]["namespace"], block)
                logger.info(
                    f'Data block {block["metadata"]["namespace"]}/{block["metadata"]["name"]} has been created '
                    f"at timestamp {int(now)}."
                )
                idx += 1
            else:
                self.timer.wait_until(start_time)

    def clean_up(self, dataset_metadata: DatasetMetadata, namespace: str):
        for data_block in dataset_metadata.data_blocks:
            block_name = data_block.name()
            self.client.delete_data_block(namespace, block_name)
            logger.info(f"Data block {namespace}/{block_name} has been deleted.")


if __name__ == "__main__":
    data_block_metadata_list = []
    data_block_metadata_list.extend(
        [
            DataBlockMetadata(
                EpsDelBudget(1, 1e-4), "taxi", "", 0, 10, release_period=10
            ),
            DataBlockMetadata(
                EpsDelBudget(1, 1e-4), "taxi", "", 0, 20, release_period=10
            ),
            DataBlockMetadata(
                EpsDelBudget(1, 1e-4), "taxi", "", 0, 30, release_period=10
            ),
        ]
    )
    dataset_metadata = DatasetMetadata("taxi", data_block_metadata_list)

    generator = BlockGenerator(PrivacyResourceClient(), Timer())
    thread = generator.create_data_blocks_periodically(
        dataset_metadata, "privacy-example"
    )
    sleep(5)
    thread.stop()

    while thread.is_alive():
        sleep(1)

    sleep(100)
    generator.clean_up(dataset_metadata, "privacy-example")
