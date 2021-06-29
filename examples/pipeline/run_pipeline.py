"""
Example pipeline to show how to use PrivateKube.
Runs the DP training and testing of an LSTM for product classification 
over the Amazon Reviews dataset.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import kfp
from kfp import dsl
from kfp import gcp
from kfp.components import func_to_container_op
from kubernetes.config import load_kube_config

from privatekube.experiments.utils import load_yaml
import privatekube

from typing import NamedTuple
from loguru import logger


# PrivateKube wrappers
allocate_blocks_op = privatekube.kfp.components.allocate_claim()
consume_blocks_op = privatekube.kfp.components.commit_claim()

# I/O utils
download_data_op = privatekube.kfp.components.download_files_from_uris()
upload_data_op = privatekube.kfp.components.upload_file_with_name()

# Load our custom DP code
def load_component(name):
    return kfp.components.load_component_from_file(
        str(Path(__file__).parent.joinpath(name).joinpath("component.yaml"))
    )


dp_preprocess_op = load_component("dp_preprocess")
dp_train_op = load_component("dp_train")
dp_evaluate_op = load_component("dp_evaluate")


# Load the secrets and configuration variables
secrets = load_yaml(Path(__file__).parent.joinpath("client_secrets.yaml"))
cluster_name = secrets["cluster_name"]
zone = secrets["zone"]

# Skip the Kubeflow cache for components with side effects (e.g. claim wrappers)
def no_cache(task):
    task.execution_options.caching_strategy.max_cache_staleness = "P0D"


# Build the pipeline
@dsl.pipeline(
    name="DP-LSTM pipeline",
    description="Preprocess (counts per category), train the DP-LSTM and DP evaluate.",
)
def pipeline(
    dataset,
    namespace,
    start_time,
    end_time,
    n_blocks,
    epsilon,
    delta,
    max_grad_norm,
    device,
    gcs_prefix,
    max_text_len,
    vocab_size,
    model,
    embedding_dim,
    hidden_dim_1,
    hidden_dim_2,
    hidden_dim,
    dropout,
    task,
    learning_rate,
    dp,
    user_level,
    n_epochs,
    batch_size,
    noise,
    timeframe_days,
    learning_rate_scheduler,
    n_workers,
):
    allocate = allocate_blocks_op(
        dataset,
        namespace,
        start_time,
        end_time,
        epsilon,
        delta,
        n_blocks,
        timeout=500,
        cluster_name=cluster_name,
        zone=zone,
    )
    no_cache(allocate)

    download = download_data_op(allocate.outputs["data"])

    dp_preprocess = dp_preprocess_op(
        input_data=download.outputs["output_path"],
        epsilon=epsilon,
        delta=delta,
        epsilon_fraction=0.1,
        delta_fraction=0,
    )
    dp_train = dp_train_op(
        n_blocks=n_blocks,
        epsilon=epsilon,
        delta=delta,
        epsilon_fraction=0.8,
        delta_fraction=1.0,
        max_grad_norm=max_grad_norm,
        device=device,
        max_text_len=max_text_len,
        vocab_size=vocab_size,
        model=model,
        embedding_dim=embedding_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task=task,
        learning_rate=learning_rate,
        dp=dp,
        user_level=user_level,
        n_epochs=n_epochs,
        batch_size=batch_size,
        noise=noise,
        timeframe_days=timeframe_days,
        learning_rate_scheduler=learning_rate_scheduler,
        n_workers=n_workers,
        block_counts=dp_preprocess.outputs["block_counts"],
        dataset_monofile=dp_preprocess.outputs["dataset_monofile"],
    )

    dp_evaluate = dp_evaluate_op(
        n_blocks=n_blocks,
        epsilon=epsilon,
        delta=delta,
        epsilon_fraction=0.1,
        delta_fraction=0.0,
        max_grad_norm=max_grad_norm,
        device=device,
        max_text_len=max_text_len,
        vocab_size=vocab_size,
        model=model,
        embedding_dim=embedding_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        hidden_dim=hidden_dim,
        dropout=dropout,
        task=task,
        learning_rate=learning_rate,
        dp=dp,
        user_level=user_level,
        n_epochs=n_epochs,
        batch_size=batch_size,
        noise=noise,
        timeframe_days=timeframe_days,
        learning_rate_scheduler=learning_rate_scheduler,
        n_workers=n_workers,
        block_counts=dp_preprocess.outputs["block_counts"],
        dataset_monofile=dp_preprocess.outputs["dataset_monofile"],
        model_path=dp_train.outputs["model_path"],
    )

    # Consume all the budget
    # We could also do partial consume requests, one for each component.
    consume = (
        consume_blocks_op(
            claim=allocate.outputs["claim"],
            budget=allocate.outputs["budget"],
            timeout=500,
            cluster_name="pkf3b",
            zone="us-east1-b",
        )
        .after(dp_preprocess)
        .after(dp_train)
        .after(dp_evaluate)
    )
    no_cache(consume)

    upload_data_op(
        dp_preprocess.outputs["statistics"], gcs_prefix, "statistics.yaml"
    ).after(consume)
    upload_data_op(dp_train.outputs["model_path"], gcs_prefix, "model.pt").after(
        consume
    )
    upload_data_op(
        dp_evaluate.outputs["metrics_path"], gcs_prefix, "metrics.yaml"
    ).after(consume)


# Connect to the API and run the pipeline

run_name = "dplstm_" + datetime.now().strftime("%m-%d-%H-%M-%S")

client = kfp.Client(
    host=secrets["host"],
    client_id=secrets["client_id"],
    other_client_id=secrets["other_client_id"],
    other_client_secret=secrets["other_client_secret"],
)
logger.info("Connected to the Kubeflow client.")

arguments = load_yaml(Path(__file__).parent.joinpath("arguments.yaml"))
arguments["gcs_prefix"] = secrets["gcs_prefix"] + f"/{run_name}"

run = client.create_run_from_pipeline_func(
    pipeline,
    arguments=arguments,
    run_name=run_name,
    experiment_name="DPLSTM pipeline demo",
    namespace=secrets["namespace"],
)
logger.info(f"Submitted the pipeline run: {run}")
