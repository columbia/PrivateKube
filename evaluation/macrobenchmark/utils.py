from loguru import logger
from pathlib import Path
from tqdm import tqdm
from privatekube.experiments.utils import save_yaml, load_yaml, yaml_dir_to_df
import numpy as np
import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import shutil

pio.templates.default = "plotly_white"

SEMANTICS = ["event", "user-time", "user"]
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.joinpath("data")
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.joinpath("logs")
DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_PYTHON_PATH = (
    Path(__file__).resolve().parent.joinpath("workload").joinpath("models")
)
DEFAULT_PARAMS_PATH = (
    Path(__file__).resolve().parent.joinpath("workload").joinpath("params")
)
DEFAULT_SELECTED_RUNS_PATH = (
    Path(__file__).resolve().parent.joinpath("workload").joinpath("runs")
)
DEFAULT_GO_EXEC = (
    Path(__file__).resolve().parent.joinpath("scheduling").joinpath("scheduling")
)
DEFAULT_EXPERIMENTS_RESULTS = DEFAULT_LOG_PATH.joinpath("amazon").joinpath(
    "classification"
)

DEFAULT_IMAGE_PATH = (
    Path(__file__).resolve().parent.joinpath("graphs").joinpath("models")
)

SEMANTICS = {
    "event": "user_level == 0",
    "user-time": "user_level == 1 and timeframe_days == 1",
    "user": "user_level == 1 and timeframe_days == 0",
}


def round_smart(f):
    if abs(f) < 0.075:
        return np.round(f, 2)
    else:
        return np.round(f, 1)


def get_name(run):
    e = round_smart(run["epsilon"])
    eps = 5.0 if e == 4.9 else e
    if "task" in run:
        task = run["task"]
    else:
        task = "stats"
    model = run["model"]
    return f"{task}-{model}-{eps}"


def copy_and_rename(file_path, destination_dir):
    d = load_yaml(file_path)
    name = get_name(d)
    shutil.copy(file_path, os.path.join(destination_dir, f"{name}.yaml"))
    return


def load_exps(list_dir, base_path=None):
    if base_path is not None:
        l = [os.path.join(base_path, d) for d in list_dir]
    else:
        l = list_dir
    return pd.concat(list(map(yaml_dir_to_df, l)))


def get_semantic(df):
    if df["user_level"][0] == 0:
        return "event"
    if df["timeframe_days"][0] == 1:
        return "user-time"
    else:
        return "user"


def get_plot_bounds(df):
    if df["task"][0] == "product":
        return 0.41, 0.4, 0.9
    if df["task"][0] == "sentiment":
        return 0.65, 0.6, 0.9


def build_gnuplot_df(df, non_private, semantic):
    naive, _, _ = get_plot_bounds(df)

    data = {
        "n_blocks": [],
        "naive_baseline": [],
        "non_dp": [],
        f"{semantic}_0.5": [],
        f"{semantic}_1.0": [],
        f"{semantic}_5.0": [],
        "n_reviews": [],
    }
    l = df.query("epsilon > 0").sort_values(["train_size", "epsilon"])
    for n_blocks in l["n_blocks"].unique():
        data["n_blocks"].append(n_blocks // 100)
        data["naive_baseline"].append(naive)
        data["non_dp"].append(
            non_private.query(f"n_blocks=={n_blocks}")["accuracy"].iloc[0]
        )
        for eps in [0.5, 1.0, 5.0]:
            acc = l.query(f"n_blocks=={n_blocks} and epsilon=={eps}")["accuracy"].iloc[
                0
            ]
            data[f"{semantic}_{eps}"].append(acc)
        data["n_reviews"].append(l.query(f"n_blocks=={n_blocks}")["train_size"].iloc[0])
    return pd.DataFrame(data=data)


def build_figure_nn(df, non_private, semantic):
    """
    Dataframe with one semantic and one model
    """

    l = df.query("epsilon > 0").sort_values(["train_size", "epsilon"])
    naive, low, high = get_plot_bounds(df)
    fig = px.line(
        l,
        x="train_size",
        y="accuracy",
        range_y=[low, high],
        color="epsilon",
        hover_data=["n_blocks", "delta", "noise"],
        title=f"{list(l['task'])[0]} {list(l['model'])[0]} {semantic} accuracy",
        log_y=False,
    ).update_traces(mode="lines+markers")
    fig.add_trace(
        go.Scatter(
            x=non_private.sort_values("train_size")["train_size"],
            y=non_private.sort_values("train_size")["accuracy"],
            mode="lines+markers",
            name="Non private",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=l["train_size"],
            y=[naive] * len(l),
            mode="lines",
            name="Naive baseline",
        )
    )
    return fig


def load_block_claims(log_claims, log_blocks, failure_ratio=0.05):
    with open(log_claims, "r") as f:
        claims = json.load(f)
    with open(log_blocks, "r") as f:
        blocks = json.load(f)
    block_interval = int(blocks[0]["metadata"]["annotations"]["blockIntervalDuration"])
    first_start_time = int(claims[0]["metadata"]["annotations"]["actualStartTime"])
    # We don't plot RDP epsilons here
    epsdel = "epsDel" in blocks[0]["status"]["availableBudget"]
    remaining_unlocked_budget = []
    remaining_locked_budget = []
    n_pipelines = []
    name = []
    empty_blocks = 0
    for b in blocks:
        try:
            if epsdel:
                unlocked = b["status"]["availableBudget"]["epsDel"]["epsilon"]
                locked = b["status"]["pendingBudget"]["epsDel"]["epsilon"]
            else:
                unlocked = [
                    r["epsilon"] for r in b["status"]["availableBudget"]["renyi"]
                ]
                locked = [r["epsilon"] for r in b["status"]["pendingBudget"]["renyi"]]

            remaining_unlocked_budget.append(unlocked)
            remaining_locked_budget.append(locked)
        except KeyError:
            remaining_unlocked_budget.append(None)
            remaining_locked_budget.append(None)
            empty_blocks += 1

        block_index = int(b["metadata"]["name"].split("-")[1])
        block_name = f"block-{block_index:02}"
        name.append(block_name)
        try:
            n_pipelines.append(len(b["status"]["acquiredBudgetMap"]))
        except KeyError:
            n_pipelines.append(0)

    if empty_blocks > failure_ratio * len(blocks):
        raise Exception(
            f"There are too many empty blocks: {empty_blocks}/{len(blocks)}"
        )

    blocks_df = pd.DataFrame(
        data={
            "name": name,
            "n_pipelines": n_pipelines,
            "remaining_unlocked_budget": remaining_unlocked_budget,
            "remaining_locked_budget": remaining_locked_budget,
        }
    )

    name = []
    n_blocks = []
    epsilon = []
    success = []
    mice = []
    block_index = []
    arrival = []
    delay = []
    priority = []
    empty_claims = 0
    for c in claims:
        try:
            name.append(c["metadata"]["name"])
            n = (
                int(
                    c["spec"]["requests"][0]["allocateRequest"]["conditions"][1][
                        "numericValue"
                    ]
                )
                - int(
                    c["spec"]["requests"][0]["allocateRequest"]["conditions"][0][
                        "numericValue"
                    ]
                )
                + 1
            )
            n_blocks.append(n)
            epsilon.append(
                c["spec"]["requests"][0]["allocateRequest"]["expectedBudget"][
                    "constant"
                ]["epsDel"]["epsilon"]
                if epsdel
                else [
                    r["epsilon"]
                    for r in c["spec"]["requests"][0]["allocateRequest"][
                        "expectedBudget"
                    ]["constant"]["renyi"]
                ]
            )
            success.append(c["status"]["responses"][0]["state"] == "success")
            mice.append("stats" in c["metadata"]["name"])
            arrival.append(
                (
                    int(c["metadata"]["annotations"]["actualStartTime"])
                    - first_start_time
                )
                / block_interval
            )
            block_index.append(
                int(
                    c["spec"]["requests"][0]["allocateRequest"]["conditions"][1][
                        "numericValue"
                    ]
                )
                + 1
            )
            delay.append(
                (
                    c["status"]["responses"][0]["allocateResponse"]["finishTime"]
                    - int(c["metadata"]["annotations"]["actualStartTime"])
                )
                / block_interval
            )
            priority.append(
                    c["spec"]["priority"]
            )
        except KeyError:
            empty_claims += 1

    if empty_claims > failure_ratio * len(claims):
        raise Exception(
            f"There are too many empty claims: {empty_claims}/{len(claims)}"
        )

    claims_df = pd.DataFrame(
        data={
            "name": name,
            "n_blocks": n_blocks,
            "epsilon": epsilon,
            "success": success,
            "mice": mice,
            "arrival": arrival,
            "delay": delay,
            "priority": priority,
        }
    )
    claims_df["size"] = np.log(
        1 + claims_df["n_blocks"] * (claims_df["epsilon"] if epsdel else 1)
    )
    claims_df["mice_text"] = claims_df["mice"].map(
        lambda b: "mice" if b else "elephants"
    )
    claims_df["success_text"] = claims_df["success"].map(
        lambda b: "success" if b else "failure"
    )

    blocks_df = blocks_df.sort_values("name")
    claims_df = claims_df.sort_values("arrival")
    return (blocks_df, claims_df)


def plot_workload_run(blocks_df, claims_df, output_dir, failure_delay=5):
    # The delay depends on N and the workload

    fig = px.scatter(
        claims_df,
        x="arrival",
        y="delay",
        color="success_text",
        symbol="mice_text",
        symbol_map={"mice": "diamond", "elephants": "circle"},
        color_discrete_map={
            "success": "royalblue",
            "failure": "firebrick",
        },
        size="size",
        hover_data=["name"],
    )

    fig.update_layout(
        title_text="Scheduling delay over time",
        xaxis_title_text="Arrival time (block index)",
        yaxis_title_text="Scheduling delay (in number of blocks)",
    )
    fig.write_image(str(output_dir.joinpath("timeline.png")))

    success_delay = claims_df[["success", "delay"]].apply(
        lambda row: row["delay"] if row["success"] else failure_delay, axis=1
    )

    pdf, bin_edges = np.histogram(
        success_delay, bins=np.linspace(0, failure_delay, num=1_000)
    )

    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]

    cdf_data = pd.DataFrame(data={"success_delay": bin_edges[:-1], "cdf": cdf})

    fig = px.line(cdf_data, x="success_delay", y="cdf", range_y=[0, 1])

    fig.update_layout(
        title_text="Cumulative density function for the scheduling delay",
        xaxis_title_text="Scheduling delay (in number of blocks)",
        yaxis_title_text="Fraction of pipelines (CDF)",
    )
    fig.write_image(str(output_dir.joinpath("delay_cdf.png")))
    cdf_data.to_csv(str(output_dir.joinpath("delay_cdf_data.csv")), index=False)


def plot_schedulers_run(metrics_df, output_dir):

    mode = "N"
    if "mode" in metrics_df and metrics_df["mode"][0] == "T":
        # Assuming a single run has only one mode at a time
        mode = "T"
    m = metrics_df.sort_values([mode])
    fig = px.line(
        m,
        x=mode,
        y="n_allocated_pipelines",
        range_y=[0, 5_000],
    )
    fig.update_layout(
        title_text=f"Number of allocated pipelines depending on the {mode} parameter",
        xaxis_title_text=f"DPF's {mode} parameter",
        yaxis_title_text="Number of allocated pipelines",
    )
    fig.write_image(str(output_dir.joinpath("allocation.png")))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=m[mode],
            y=m["n_allocated_mice"],
            name="Allocated mice",
            line=dict(color="green"),
            stackgroup="one",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m[mode],
            y=m["n_allocated_elephants"],
            name="Allocated lephants",
            line=dict(color="blue"),
            stackgroup="one",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m[mode],
            y=m["n_mice"] - m["n_allocated_mice"],
            name="Unallocated mice",
            line=dict(color="lightgreen"),
            stackgroup="one",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m[mode],
            y=m["n_elephants"] - m["n_allocated_elephants"],
            name="Unallocated elephants",
            line=dict(color="lightblue"),
            stackgroup="one",
        )
    )
    fig.update_layout(
        title_text=f"Number of allocated mice/elephants depending on the {mode} parameter",
        xaxis_title_text=f"DPF's {mode} parameter",
        yaxis_title_text="Number of pipelines",
    )
    fig.update_yaxes(range=[0, 5_000])
    fig.write_image(str(output_dir.joinpath("mice_elephants_allocation.png")))
