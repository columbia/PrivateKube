import typer
from typing import Optional
from loguru import logger

from privatekube.platform.privacy_resource_client import PrivacyResourceClient
from privatekube.platform.privacy_budget import (
    PrivacyBudget,
    EpsDelBudget,
)
from privatekube.platform.block import make_time_partitioned_data_block

ROOT = "gs://privatekube-public-artifacts/data/blocks"
INITIAL_BUDGET = EpsDelBudget(epsilon=10, delta=1e-7)
NAMESPACE = "privacy-example"
DATASET = "amazon"

app = typer.Typer()


def main(create: bool, days: int, users: int):
    client = PrivacyResourceClient()
    for day in range(days):
        for user_bucket in range(users):
            blockname = f"{day}-{user_bucket}"

            if create:
                datasource = f"{ROOT}/{blockname}.h5"

                block = make_time_partitioned_data_block(
                    name=blockname,
                    namespace=NAMESPACE,
                    dataset=DATASET,
                    start_time=day,
                    end_time=day,
                    data_source=datasource,
                    initial_budget=INITIAL_BUDGET,
                    release_duration=None,
                )
                client.create_data_block(NAMESPACE, block)

                logger.info(f"Data block {NAMESPACE}/{blockname} has been created.")
            else:
                client.delete_data_block(NAMESPACE, blockname)
                logger.info(f"Data block {NAMESPACE}/{blockname} has been deleted.")


@app.command()
def create(days: int = typer.Option(50, help="Number of days"), users: int=typer.Option(100, help="Number of user buckets between 1 and 100")):
    main(create=True, days=days, users=users)


@app.command()
def delete(days: int = typer.Option(50, help="Number of days"), users: int=typer.Option(100, help="Number of user buckets between 1 and 100")):
    main(create=False, days=days, users=users)


if __name__ == "__main__":
    app()