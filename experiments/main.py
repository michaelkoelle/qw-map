"""Quantum Supervised Learning Testbed"""


import dataclasses
import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, MeanMetric

import wandb
from config import Config, Dataset, Encoding, Layers, ShapingFunction
from plots import create_conf_matrix, create_csv, plot_experiment, save_test_csv
from vqc import VQC

iris_hyparams: Config = Config(
    model="VQC",
    dataset=Dataset.IRIS.value,
    seed_value=0,
    lr=0.020053112845816387,
    epochs=50,
    batch_size=9,
    encoding=Encoding.ANGLE_ENCODING_X.value,
    layers=Layers.STRONGLY_ENTANGLING_LAYERS.value,
    weight_decay=0.03719775843379234,
    num_layers=8,
    shaping_function=ShapingFunction.NONE.value,
    batch_norm=False,
    data_reuploading=False,
)

wine_hyparams: Config = Config(
    model="VQC",
    dataset=Dataset.WINE.value,
    seed_value=0,
    lr=0.08553066703887986,
    epochs=100,
    batch_size=13,
    encoding=Encoding.ANGLE_ENCODING_Y.value,
    layers=Layers.STRONGLY_ENTANGLING_LAYERS.value,
    weight_decay=0.0009267887893905224,
    num_layers=12,
    shaping_function=ShapingFunction.NONE.value,
    batch_norm=False,
    data_reuploading=False,
)

breast_cancer_hyparams: Config = Config(
    model="VQC",
    dataset=Dataset.BREAST_CANCER.value,
    seed_value=0,
    lr=0.09321207318215433,
    epochs=100,
    batch_size=14,
    encoding=Encoding.AMPLITUDE_ENCODING.value,
    layers=Layers.STRONGLY_ENTANGLING_LAYERS.value,
    weight_decay=0.033675237827537334,
    num_layers=7,
    shaping_function=ShapingFunction.NONE.value,
    batch_norm=False,
    data_reuploading=False,
)


def main():
    """Entrypoint of the script"""
    run_count = 10
    config_defaults: Config = iris_hyparams
    data = []
    now = datetime.datetime.now()
    time = now.strftime("%Y%m%d-%H%M%S")
    out_dir = "output/" + time + "/"

    data.extend(exec_exp(config_defaults, list(range(run_count)), out_dir))

    df, _ = create_csv(data, out_dir)
    plot_experiment(df, out_dir=Path(out_dir))


def exec_exp(conf: Config, seeds: List[int], out_dir: str) -> List[Dict[str, Any]]:
    """Runs one experiment consisting of multiple runs, given by the seed range"""
    data: List[Dict[str, Any]] = []
    y_true_list = []
    y_preds_list = []
    labels_list = []

    for index, seed in enumerate(seeds):
        print("Computing run n. ", index)
        conf.seed_value = seed
        run_data, y_true, y_preds, labels = exec_run(conf)
        data.extend(run_data)
        y_true_list.extend(y_true)
        y_preds_list.extend(y_preds)
        labels_list = labels

    df, df_labels = save_test_csv(
        y_true=y_true_list,
        y_preds=y_preds_list,
        labels=labels_list,
        out_dir=out_dir,
        file_name="data_reup-"
        + str(conf.data_reuploading)
        + "-"
        + conf.dataset
        + "-"
        + conf.shaping_function
        + "-",
    )

    create_conf_matrix(
        df,
        list(df_labels["labels"]),
        out_dir=out_dir,
        file_name="data_reup-"
        + str(conf.data_reuploading)
        + "-"
        + conf.dataset
        + "-"
        + conf.shaping_function
        + "-cf",
    )

    return data


def set_seed(seed_value: int):
    """Sets the seed"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def exec_run(config_defaults: Config):
    """Executes one run"""

    run_data = []

    run = wandb.init(
        project="quantum-weight-shaping",
        entity="mdsg",
        config=dataclasses.asdict(config_defaults),
        reinit=True,
    )

    if run is None:
        raise Exception("Run initialization failed!")

    config: Config = wandb.config
    print(config)
    dataset = Dataset.get_instance(config.dataset)
    num_qubits = Encoding.get_num_qubits(
        config.encoding, dataset.num_classes(), dataset.num_features()
    )
    run.summary["num_qubits"] = num_qubits

    # for a VQC
    model = VQC(
        dataset.classes(),
        num_qubits,
        Encoding.get_instance(config.encoding),
        Layers.get_instance(config.layers),
        ShapingFunction.get_instance(config.shaping_function),
        config.num_layers,
        config.batch_norm,
        config.data_reuploading,
    )

    # for a classical NN
    # model = NeuralNetwork(
    #     dataset.classes(),
    #     dataset.num_features(),
    #     config.num_layers,
    # )

    # Track model
    wandb.watch(model)

    # ! Only for Sweeps, Random seed
    # seed = np.random.randint(2**32)
    # wandb.config.update({"seed_value": seed}, allow_val_change=True)
    # ! ###############

    set_seed(config.seed_value)

    # Datasets
    train_split = 0.8
    train_size = int(train_split * len(dataset))
    valid_size = int((1 - train_split) / 2 * len(dataset))

    train_d, valid_d, test_d = random_split(dataset, [train_size, valid_size + 1, valid_size + 1])

    train_dl = DataLoader(train_d, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_d, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_d, batch_size=config.batch_size, shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), weight_decay=config.weight_decay, lr=config.lr)

    max_acc_valid = 0
    max_acc_step = 0
    max_acc_state_dict = None

    for epoch in range(config.epochs):
        epoch_data = dict(wandb.config)
        # Threshold: classifier needs to be calibrated
        acc_train = Accuracy(threshold=0.5, num_classes=dataset.num_classes())
        acc_valid = Accuracy(threshold=0.5, num_classes=dataset.num_classes())
        loss_train = MeanMetric()
        loss_valid = MeanMetric()

        # Train
        model.train()

        for train_batch in train_dl:
            x, y = train_batch
            y_pred = model(x)
            # model.draw(x)

            loss = criterion(y_pred, y.long())

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_train.update(loss.item())
            acc_train.update(y_pred, y.long())

        # Validation
        with torch.no_grad():
            model.eval()
            for valid_batch in valid_dl:
                x, y = valid_batch
                y_pred = model(x.float())
                loss = criterion(y_pred, y.long())
                loss_valid.update(loss.item())
                acc_valid.update(y_pred, y.long())
            model.train()

        if acc_valid.compute().item() >= max_acc_valid:
            max_acc_valid = acc_valid.compute().item()
            max_acc_state_dict = model.state_dict()
            max_acc_step = epoch

        print(
            f"Epoch: {(epoch+1):5d} | "
            f"Loss_train: {loss_train.compute().item():0.7f} | "
            f"Acc_train: {acc_train.compute().item():0.7f} | "
            f"Loss_valid: {loss_valid.compute().item():0.7f} | "
            f"Acc_valid: {acc_valid.compute().item():0.7f}"
        )

        epoch_data.update(
            {
                "step": epoch,
                "loss_train": loss_train.compute().item(),
                "acc_train": acc_train.compute().item(),
                "acc_valid": acc_valid.compute().item(),
                "loss_valid": loss_valid.compute().item(),
            }
        )

        wandb.log(
            {
                "loss_train": loss_train.compute().item(),
                "acc_train": acc_train.compute().item(),
                "acc_valid": acc_valid.compute().item(),
                "loss_valid": loss_valid.compute().item(),
            }
        )

        run_data.append(epoch_data)

    # Test late
    # Load best param config
    if max_acc_state_dict is not None:
        model.load_state_dict(max_acc_state_dict)
    acc_test = Accuracy(threshold=0.5, num_classes=dataset.num_classes())
    loss_test = MeanMetric()
    probs = []
    preds = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for test_batch in test_dl:
            x, y = test_batch
            y_pred = model(x.float())
            loss = criterion(y_pred, y.long())

            y_true.extend(y.long().tolist())
            probs.extend(y_pred.tolist())
            preds.extend([np.argmax(y).item() for y in y_pred])

            loss_test.update(loss.item())
            acc_test.update(y_pred, y.long())
        model.train()

    print(
        f"Loss_test: {loss_test.compute().item():0.7f} | "
        f"Acc_test {acc_test.compute().item():0.7f} | "
        f"Step: {max_acc_step} | "
    )

    run.summary["acc_test"] = acc_test.compute().item()
    run.summary["loss_test"] = loss_test.compute().item()
    run.summary["test_step"] = max_acc_step

    for row in run_data:
        row.update(
            {
                "acc_test": acc_test.compute().item(),
                "loss_test": loss_test.compute().item(),
                "test_step": max_acc_step,
            }
        )

    wandb.log(
        {
            "conf_mat_late": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=preds,
                class_names=dataset.class_names(),
            ),
            "roc_late": wandb.plot.roc_curve(
                y_true=y_true, y_probas=probs, labels=dataset.class_names()
            ),
            "pr_late": wandb.plot.pr_curve(
                y_true=y_true, y_probas=probs, labels=dataset.class_names()
            ),
        }
    )

    run.summary["sum_train_params"] = sum(
        [np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]  # type: ignore
    )

    for row in run_data:
        row.update(
            {
                "sum_train_params": sum(
                    [
                        np.prod(p.size())
                        for p in filter(lambda p: p.requires_grad, model.parameters())  # type: ignore
                    ]
                )
            }
        )

    run.finish()

    return run_data, y_true, preds, dataset.class_names()


if __name__ == "__main__":
    main()
