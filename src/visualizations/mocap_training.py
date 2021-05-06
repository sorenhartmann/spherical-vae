from pathlib import Path
import pandas as pd
import seaborn as sns

run_dir = Path(__file__).parents[2] / "runs"
overleaf_dir = Path(__file__).resolve().parents[2] / "overleaf" / "figures"
sns.set_theme("talk", style="whitegrid")

if __name__ == "__main__":

    experiments = [
        ("Swimming", "swimming"),
        ("Walking", "walk-walk"),
        ("Run/walk", "run-walk"),
        ("Dancing", "dancing"),
    ]
    models = [("SVAE", "svae"), ("VAE", "vae")]

    df = pd.DataFrame(
        columns=["Epoch", "Neg. $\\beta$-ELBO", "Neg. ELBO", "KL-term", "Experiment", "Model"]
    )

    for exp_name, exp_id in experiments:

        exp_dir = run_dir / "mocap-training" / exp_id

        for model_name, model_id in models:

            loss_data = pd.read_csv(exp_dir / f"run-{model_id}_0-tag-Loss_Train.csv")
            elbo_data = pd.read_csv(
                exp_dir / f"run-{model_id}_0-tag-Loss_Validation.csv"
            )
            kl_data = pd.read_csv(exp_dir / f"run-{model_id}_0-tag-Average KL-term.csv")

            df = df.append(
                pd.DataFrame(
                    {
                        "Epoch": loss_data["Step"],
                        "Neg. $\\beta$-ELBO": loss_data["Value"],
                        "Neg. ELBO": elbo_data["Value"],
                        "KL-term": kl_data["Value"],
                        "Experiment": exp_name,
                        "Model": model_name,
                    }
                )
            )

    sns.relplot(
        data=df[df["Epoch"] > 50],
        x="Epoch",
        y="Neg. $\\beta$-ELBO",
        hue="Model",
        kind="line",
        col="Experiment",
        col_wrap=2
    ).savefig(overleaf_dir / "mocap_training_loss.pdf")

    sns.relplot(
        data=df[(df["Epoch"] > 50) & (df["Neg. ELBO"] < 800)],
        x="Epoch",
        y="Neg. ELBO",
        hue="Model",
        kind="line",
        col="Experiment",
        col_wrap=2,
        facet_kws={ "sharey" : False},
    ).savefig(overleaf_dir / "mocap_val_elbo.pdf")

    sns.relplot(
        data=df[df["Epoch"] > 50],
        x="Epoch",
        y="KL-term",
        hue="Model",
        kind="line",
        col="Experiment",
        col_wrap=2
    ).savefig(overleaf_dir / "mocap_val_kl_term.pdf")

