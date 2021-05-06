import os
from pathlib import Path
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

run_dir = Path(__file__).parents[2] / "runs"
overleaf_dir = Path(__file__).resolve().parents[2] / "overleaf" / "figures"
sns.set_theme("talk", style="whitegrid")

def tabulate_events(dpath):
    summary_iterators = [
        EventAccumulator(os.path.join(dpath, dname)).Reload()
        for dname in os.listdir(dpath)
    ]

    tags = summary_iterators[0].Tags()["scalars"]

    for it in summary_iterators:
        assert it.Tags()["scalars"] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps


def to_csv(dpath):
    dirs = os.listdir(dpath)

    d, steps = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values = np.array(values, dtype=object)

    dfs_out = {}
    for index, tag in enumerate(tags):
        dfs_out[tag] = pd.DataFrame(np_values[index], columns=dirs)
        # df = pd.DataFrame(np_values[index], index=steps, columns=dirs)

    return dfs_out


def get_file_path(dpath, tag):
    file_name = tag.replace("/", "_") + ".csv"
    folder_path = os.path.join(dpath, "csv")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == "__main__":
    path = run_dir / "gradients"
    dfs = to_csv(path)

    validation_df = pd.DataFrame(columns=["Neg. ELBO", "KL-term", "Corrected", "Run"])
    training_df = pd.DataFrame(columns=["Neg. ELBO", "Corrected", "Run"])

    for run in dfs["Loss/Train"]:

        run_id = int(run.split("_")[-1])
        corrected = "wo" not in run

        training_df = training_df.append(
            pd.DataFrame(
                {
                    "Loss ($\\beta$-ELBO)": dfs["Loss/Train"][run],
                    "Corrected": corrected,
                    "Run": run_id,
                }
            )
        )

    for run in dfs["Loss/Validation"]:

        run_id = int(run.split("_")[-1])
        corrected = "wo" not in run

        validation_df = validation_df.append(
            pd.DataFrame(
                {
                    "Neg. ELBO": dfs["Loss/Validation"][run],
                    "KL-term": dfs['Average KL-term'][run],
                    "Corrected": corrected,
                    "Run": run_id,
                }
            )
        )

    training_df.reset_index(inplace=True)
    validation_df.reset_index(inplace=True)

    fig = sns.relplot(
        data=training_df,
        y="Loss ($\\beta$-ELBO)",
        x="index",
        kind="line",
        units="Run",
        estimator=None,
        aspect=2,
        hue="Corrected",
        hue_order=[True, False]
    )

    if overleaf_dir.exists():
        fig.savefig(overleaf_dir / "training_loss_gradient_corr.pdf")
        
    fig = sns.relplot(
        data=validation_df,
        y="Neg. ELBO",
        x="index",
        kind="line",
        units="Run",
        estimator=None,
        aspect=2, 
        hue="Corrected",
        hue_order=[True, False]
    )

    fig.savefig(overleaf_dir / "validation_loss_gradient_corr.pdf")

    fig = sns.relplot(
        data=validation_df,
        y="KL-term",
        x="index",
        kind="line",
        units="Run",
        estimator=None,
        aspect=2,
        hue="Corrected",
        hue_order=[True, False]
    )

    fig.savefig(overleaf_dir / "KL_term_gradient_corr.pdf")



