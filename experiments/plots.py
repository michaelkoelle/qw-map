"""Creating paper plots"""
import itertools
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

plt.rcParams["text.usetex"] = True
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
sns.color_palette("colorblind")

palette = itertools.cycle(sns.color_palette("colorblind"))  # type: ignore


def plot_shaping_functions(out_dir: Path = Path("output/functions/")):
    """Plot shaping functions"""

    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-4, 4, 200)
    sns.set(font_scale=3.5)
    sns.set_style("whitegrid")

    # Function definitions
    data = {
        "x": x,
        "w/o weight constraint": x,
        "clamp": np.maximum(np.minimum(x, np.pi), -np.pi),
        "tanh": np.pi * np.tanh(x),
        "arctan": 2.0 * np.arctan(2 * x),
        "sigmoid": 2 * np.pi * (1 / (1 + np.power(np.e, -x))) - np.pi,
        "elu": [x_i if x_i > 0 else np.pi * (np.power(np.e, x_i) - 1) for x_i in x],
    }

    all_data = {"mapping function": [], "x": [], "y": []}

    for key, value in data.items():
        if key == "x":
            continue
        for i, v in enumerate(value):
            all_data["mapping function"].append(key)
            all_data["x"].append(x[i])
            all_data["y"].append(v)

    df_all = pd.DataFrame.from_dict(all_data)
    df = pd.DataFrame.from_dict(data)

    fig = sns.lineplot(data=df_all, x="x", y="y", hue="mapping function", zorder=2, linewidth=3)
    fig.axhline(np.pi, linestyle="dashed", color="grey", zorder=1)
    fig.axhline(-np.pi, linestyle="dashed", color="grey", zorder=1)
    fig.axvline(np.pi, linestyle="dashed", color="grey", zorder=1)
    fig.axvline(-np.pi, linestyle="dashed", color="grey", zorder=1)
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.xticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\\pi$", "$-\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$", "$\\pi$"],
    )
    plt.yticks(
        [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
        ["$-\\pi$", "$-\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$", "$\\pi$"],
    )
    plt.xlabel("Parameter value")
    plt.ylabel("Mapped output")
    # plt.setp(fig.get_legend().get_texts(), fontsize="12")  # for legend text
    # plt.setp(fig.get_legend().get_title(), fontsize="12")  # for legend title
    fig.legend(fontsize=10)
    plt.grid()
    plt.savefig(f"{str(out_dir)}/all.pdf", bbox_inches="tight")
    plt.clf()

    for func_name in data:
        if func_name == "x":
            continue
        fig = sns.lineplot(data=df, x="x", y=func_name, zorder=2, linewidth=6, color=next(palette))
        fig.axhline(np.pi, linestyle="dashed", color="grey", zorder=1)
        fig.axhline(-np.pi, linestyle="dashed", color="grey", zorder=1)
        fig.axvline(np.pi, linestyle="dashed", color="grey", zorder=1)
        fig.axvline(-np.pi, linestyle="dashed", color="grey", zorder=1)
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])
        plt.xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            ["$-\\pi$", "$-\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$", "$\\pi$"],
        )
        plt.yticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            ["$-\\pi$", "$-\\frac{\\pi}{2}$", "0", "$\\frac{\\pi}{2}$", "$\\pi$"],
        )
        plt.xlabel("Parameter value")
        plt.ylabel("Mapped output")
        plt.grid(visible=True)
        test = func_name.replace("/", "")
        plt.savefig(f"{str(out_dir)}/{test}.pdf", bbox_inches="tight")
        plt.clf()


def create_csv(data: List[Dict[str, Any]], out_dir: str):
    """Create csv from experiment data"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    # Save .csv
    df.to_csv(out_dir + "data.csv", sep=",", encoding="utf-8", index=False)

    return df, out_dir + "data.csv"


def create_line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    xlabel: str,
    ylabel: str,
    out_dir: str,
    file_name: str,
):
    """Create a line plot"""
    sns.set(font_scale=1.5)
    fig = sns.lineplot(data=df, x=x, y=y, hue=hue, errorbar="sd")
    fig.legend(fontsize=10)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if "iris" in file_name:
        plt.xlim(0, 10)
    elif "wine" in file_name:
        plt.xlim(0, 50)
    elif "breast" in file_name:
        plt.xlim(0, 50)

    if "acc" in file_name:
        plt.ylim(0, 1)

    plt.savefig(out_dir + file_name + ".pdf", bbox_inches="tight")
    plt.clf()


def save_test_csv(
    y_true: List[Any], y_preds: List[Any], labels: List[Any], out_dir: str, file_name: str
):
    """Save test data to csv"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "y_true": [labels[i] for i in y_true],
            "y_preds": [labels[i] for i in y_preds],
        }
    )

    df_labels = pd.DataFrame(
        {
            "labels": labels,
        }
    )

    df.to_csv(out_dir + file_name + "data_test.csv", sep=",", encoding="utf-8", index=False)
    df_labels.to_csv(out_dir + file_name + "labels.csv", sep=",", encoding="utf-8", index=False)

    return df, df_labels


def create_conf_matrix(
    df: pd.DataFrame,
    labels: List[str],
    out_dir: str,
    file_name: str,
):
    """Creates a confusion matrix plot"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cf_matrix = confusion_matrix(y_true=df["y_true"], y_pred=df["y_preds"], labels=labels)
    norm_cf = cf_matrix / np.sum(cf_matrix)
    ax = sns.heatmap(
        norm_cf,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,  # type: ignore
        yticklabels=labels,  # type: ignore
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label")
    plt.savefig(out_dir + file_name + ".pdf", bbox_inches="tight")
    plt.clf()


def create_pr_curve(
    y_true: Any,
    y_pred: Any,
    out_dir: str,
    file_name: str,
):
    """Creates a precision recall curve"""
    RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.savefig(out_dir + file_name + ".pdf", bbox_inches="tight")
    plt.clf()


def create_roc_curve(
    y_true: Any,
    y_pred: Any,
    out_dir: str,
    file_name: str,
):
    """Creates a roc curve"""
    PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.savefig(out_dir + file_name + ".pdf", bbox_inches="tight")
    plt.clf()


def plot_experiment(df: pd.DataFrame, out_dir: Path = Path("output/plots/")):
    """Plot experiment"""
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = 0.6

    df["acc_train_ema"] = df["acc_train"].ewm(alpha=alpha).mean()
    df["acc_valid_ema"] = df["acc_valid"].ewm(alpha=alpha).mean()
    df["loss_train_ema"] = df["loss_train"].ewm(alpha=alpha).mean()
    df["loss_valid_ema"] = df["loss_valid"].ewm(alpha=alpha).mean()

    data_reups = df["data_reuploading"].unique()
    data_reup_groups = df.groupby(["data_reuploading"])

    for data_reup in data_reups:
        # data_reup_dir = str(out_dir) + "/data_reup_" + str(data_reup) + "/"
        # Path(data_reup_dir).mkdir(parents=True, exist_ok=True)
        data_reup_df = data_reup_groups.get_group(data_reup)

        datasets = data_reup_df["dataset"].unique()
        dataset_groups = data_reup_df.groupby(["dataset"])

        for dataset in datasets:
            # dataset_dir = data_reup_dir + str(dataset) + "/"
            # Path(dataset_dir).mkdir(parents=True, exist_ok=True)
            dataset_df = dataset_groups.get_group(dataset)

            # functions = dataset_df["shaping_funcion"].unique()
            # function_groups = dataset_df.groupby(["shaping_funcion"])

            create_line_plot(
                df=dataset_df,
                x="step",
                y="acc_train_ema",
                hue="shaping_function",
                xlabel="Epoch",
                ylabel="Training Accuracy",
                out_dir=str(out_dir) + "/",
                file_name="data_reup-" + str(data_reup) + "-" + dataset + "-acc-train",
            )

            create_line_plot(
                df=dataset_df,
                x="step",
                y="acc_valid_ema",
                hue="shaping_function",
                xlabel="Epoch",
                ylabel="Validation Accuracy",
                out_dir=str(out_dir) + "/",
                file_name="data_reup-" + str(data_reup) + "-" + dataset + "-acc-valid",
            )

            create_line_plot(
                df=dataset_df,
                x="step",
                y="loss_train_ema",
                hue="shaping_function",
                xlabel="Epoch",
                ylabel="Training Loss",
                out_dir=str(out_dir) + "/",
                file_name="data_reup-" + str(data_reup) + "-" + dataset + "-loss-train",
            )

            create_line_plot(
                df=dataset_df,
                x="step",
                y="loss_valid_ema",
                hue="shaping_function",
                xlabel="Epoch",
                ylabel="Validation Loss",
                out_dir=str(out_dir) + "/",
                file_name="data_reup-" + str(data_reup) + "-" + dataset + "-loss-valid",
            )

            # for function in functions:
            #     function_dir = dataset_dir + str(function) + "/"
            #     Path(function_dir).mkdir(parents=True, exist_ok=True)
            #     function_df = function_groups.get_group(function)

            #     fig = sns.lineplot(
            #         data=function_df, x="step", y="acc_valid", hue="shaping_funcion", errorbar="sd"
            #     )

            #     plt.savefig(function_dir + function + ".pdf")
            #     plt.clf()


def create_multi_plot_data_frame(df: pd.DataFrame, out_dir: Path = Path("output/plots/")):
    """Create multi plot df"""
    df_no_datareup = df[df["data_reuploading"] == False]
    df_filtered_datasets = df_no_datareup[df_no_datareup["dataset"].isin(["iris", "wine"])]
    df_filtered_func = df_filtered_datasets[
        ~df_filtered_datasets["shaping_function"].isin(["classical NN", "random"])
    ]
    df_next = df_filtered_func[["shaping_function", "dataset", "step", "acc_valid", "loss_valid"]]
    df_end = df_next.melt(
        id_vars=["shaping_function", "dataset", "step"],
        value_vars=["acc_valid", "loss_valid"],
        var_name="type",
    )
    df_end = df_end.sort_values(by="dataset")
    df_end.to_csv(str(out_dir) + "/multi_plot.csv")
    return df_end


def create_multi_plot_experiments(df: pd.DataFrame, out_dir: Path = Path("output/plots/")):
    """Create multi plot"""

    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    plt.rcParams["text.usetex"] = True
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)

    g = sns.FacetGrid(
        df,
        col="dataset",
        row="type",
        sharex=False,
        sharey=False,
        despine=True,
        height=4,
        aspect=2,
        palette=sns.color_palette("colorblind"),
    )

    g.map_dataframe(
        sns.lineplot,
        x="step",
        y="value",
        hue="shaping_function",
        errorbar="sd",
        palette=sns.color_palette("colorblind"),
        hue_order=["w/o Re-Mapping", "Clamp", "Tanh", "Arctan", "Sigmoid", "ELU"],
    )

    g.add_legend()
    sns.move_legend(g, "upper left", bbox_to_anchor=(0.84, 0.7), frameon=False)

    axes = g.axes.flatten()  # type: ignore

    for ax in axes:
        for coll in ax.collections:
            coll.set_alpha(0.1)

    axes[0].set_ylabel("Accuracy (Validation)")
    axes[0].set_title("Iris Dataset")
    axes[0].set_xlabel("")
    axes[0].set_xlim(0, 10)
    axes[0].lines[1].set_linestyle("--")

    axes[1].set_ylabel("")
    axes[1].set_title("Wine Dataset")
    axes[1].set_xlabel("")
    axes[1].set_xlim(0, 50)
    axes[1].lines[1].set_linestyle("--")

    axes[2].set_ylabel("Loss (Validation)")
    axes[2].set_title("")
    axes[2].set_xlabel("Epoch")
    axes[2].set_xlim(0, 10)
    axes[2].lines[1].set_linestyle("--")

    axes[3].set_ylabel("")
    axes[3].set_title("")
    axes[3].set_xlabel("Epoch")
    axes[3].set_xlim(0, 50)
    axes[3].lines[1].set_linestyle("--")

    plt.savefig(str(out_dir) + "/results_valid.pdf", bbox_inches="tight")
    plt.clf()


def calc_conf_int(path: str, label_path: str):
    """Calculate 95% confidence interval"""

    df_from_csv = pd.read_csv(path)

    df_labels = pd.read_csv(label_path)

    create_conf_matrix(
        df=df_from_csv,
        labels=list(df_labels["labels"]),
        out_dir=str("/Users/m.koelle/Repositories/qsl-testbed/output/plots/"),
        file_name="conf_matrix",
    )

    y_true = df_from_csv["y_true"].tolist()
    y_preds = df_from_csv["y_preds"].tolist()

    n = len(y_true)
    accuracy = sum([t == p for (t, p) in zip(y_true, y_preds)]) / n

    print(path)
    print(round(accuracy, 3))

    # ## 95% conf interval
    interval = 1.96 * math.sqrt((accuracy * (1 - accuracy)) / n)
    print(round(interval, 3))
    print("##################")


if __name__ == "__main__":
    # plot_shaping_functions()

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-none-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-hard_clamp-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-tanh-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-arctan-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-sigmoid-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    # calc_conf_int(
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/data_reup-False-iris-elu-data_test.csv",
    #     "/Users/m.koelle/Repositories/qsl-testbed/output/iris_labels.csv",
    # )

    df_csv = pd.read_csv("/Users/m.koelle/Repositories/qsl-testbed/output/data_wine_iris.csv")
    df_final = create_multi_plot_data_frame(df_csv)
    df_final = pd.read_csv("/Users/m.koelle/Repositories/qsl-testbed/output/plots/multi_plot.csv")
    create_multi_plot_experiments(df_final)

    # plot_experiment(df_csv)
    # plot_shaping_functions()

    # df = pd.read_csv("/Users/m.koelle/Repositories/qsl-testbed/output/data_wine_iris.csv")
    # df_iris = df[df["dataset"] == "iris"]

    # df_iris = df_iris[df_iris["step"] < 10]
    # df_iris = df_iris[df_iris["shaping_function"] == "ELU"]
    # df_iris = df_iris.groupby(["step"]).mean()
    # print(df_iris)
