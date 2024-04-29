import argparse
from pathlib import Path
from typing import Dict, List, Optional

import hyperopt
import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_OK, Trials, fmin, hp
from torch import Tensor

from tlc.eval import compute_metrics, evaluate_ee, evaluate_ics
from tlc.utils import load_data
from tlc.visualize import plot_cost_vs_acc, plot_heatmap, plot_loss


def decode_bias_dict(
    bias_dict: Dict, n_ics: int, classes_per_task: List[int], device: str
) -> Tensor:
    n_tasks = len(classes_per_task)

    a = bias_dict["a"]
    b = bias_dict["b"]

    dists = n_ics * [[n_tasks - 1 - i for i in range(n_tasks)]]
    dists = torch.tensor(dists)
    b = torch.ones_like(dists) * b
    b[:, -1] = 0
    bias = a * dists + b

    return bias.to(device, non_blocking=True)


def expand_bias(bias: Tensor, classes_per_task: List[int]) -> Tensor:
    if len(set(classes_per_task)) == 1:
        return torch.repeat_interleave(bias, classes_per_task[0], dim=1)
    else:
        new_bias = torch.zeros(bias.shape[0], sum(classes_per_task))
        for i in range(len(classes_per_task)):
            for j in range(classes_per_task[i]):
                cur_idx = sum(classes_per_task[:i]) + j
                new_bias[:, cur_idx : cur_idx + 1] = bias[:, i].unsqueeze(1)
        return new_bias


def optimize_tlc(
    logits: Tensor,
    classes_per_task: List[int],
    optimization_algorithm: str = "tpe",
    max_iters: int = 1000,
    hp_space: str = "normal",
    hp_mu: Optional[float] = None,
    hp_sigma: Optional[float] = None,
    hp_min: Optional[float] = None,
    hp_max: Optional[float] = None,
    seed=0,
):
    _, n_ics, n_classes = logits.shape

    if optimization_algorithm == "tpe":
        algo = hyperopt.tpe.suggest
    elif optimization_algorithm == "random":
        algo = hyperopt.rand.suggest
    else:
        raise NotImplementedError()

    device = logits.device
    if hp_space == "normal":
        hp_init = {
            "a": hp.normal("a", mu=hp_mu, sigma=hp_sigma),
            "b": hp.normal("b", mu=hp_mu, sigma=hp_sigma),
        }
    elif hp_space == "uniform":
        hp_init = {
            "a": hp.uniform("a", low=hp_min, high=hp_max),
            "b": hp.uniform("b", low=hp_min, high=hp_max),
        }
    else:
        raise NotImplementedError()

    def loss_fn(bias_dict):
        bias = decode_bias_dict(bias_dict, n_ics, classes_per_task, device)
        bias = expand_bias(bias, classes_per_task).unsqueeze(dim=0)
        adapted_logits = logits + bias
        preds = torch.argmax(adapted_logits, dim=-1)  # [batch_size, n_ics]
        pred_mask = torch.nn.functional.one_hot(
            preds, num_classes=logits.shape[-1]
        )  # [batch_size, n_ics, n_classes]
        masked_logits = torch.where(
            pred_mask == 1, adapted_logits * -torch.inf, adapted_logits
        )  # [batch_size, n_ics, n_classes]
        per_task_preds = torch.stack(
            torch.split(masked_logits, classes_per_task, dim=2), dim=-1
        )  # [batch_size, n_ics, n_classes_per_task, n_tasks]
        max_ood_logits = torch.max(
            per_task_preds, dim=-2
        ).values  # [batch_size, n_ics, n_tasks]
        mean_max_ood_logits = (
            max_ood_logits.mean(dim=-1).mean(dim=-1).unsqueeze(1).unsqueeze(2)
        )  # [batch_size, 1, 1]
        ood_mse = (max_ood_logits - mean_max_ood_logits) ** 2
        return {"loss": float(torch.mean(ood_mse)), "status": STATUS_OK}

    trials = Trials()
    best_coefficients = fmin(
        fn=loss_fn,
        space=hp_init,
        algo=algo,
        max_evals=max_iters,
        trials=trials,
        rstate=np.random.default_rng(seed),
    )
    best_bias = decode_bias_dict(best_coefficients, n_ics, classes_per_task, device)
    return best_bias, trials.losses()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_logits_glob",
        type=str,
        required=True,
        help="Directory with logits for training TLC parameters. The loss function will use all data "
        "in this directory, so to run TLC describted in the paper one should provide the "
        "directory with only last task logits.",
    )
    parser.add_argument(
        "--test_logits_glob",
        type=str,
        required=True,
        help="Directory with logits for testing. Should contain all tasks data.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=str, choices=["cifar100", "tiny", "in"], default="cifar100"
    )
    parser.add_argument(
        "--optimization_algorithm",
        type=str,
        choices=["tpe", "random"],
        default="tpe",
    )
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument(
        "--hp_space", type=str, choices=["normal", "uniform"], default="normal"
    )
    parser.add_argument("--hp_mu", type=float, default=0)
    parser.add_argument("--hp_sigma", type=float, default=5)
    parser.add_argument("--hp_min", type=float, default=-2)
    parser.add_argument("--hp_max", type=float, default=2)
    parser.add_argument("--device", type=str, default="cpu")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tlc_data = load_data(args.train_logits_glob, args.test_logits_glob)
    output_dir = Path(args.output_dir)
    for root_dir, data in tlc_data.items():
        output_path = output_dir / root_dir
        output_path.mkdir(parents=True, exist_ok=True)
        train_logits = data["train_logits_tag"]
        train_targets = data["train_targets"]
        test_logits = data["test_logits_tag"]
        test_targets = data["test_targets"]
        n_tasks = data["n_tasks"]
        classes_per_task = [d["classes_per_task"] for d in data["test_data_per_task"]]
        if len(set(classes_per_task)) != 1:
            raise NotImplementedError("Only same sized tasks are handled for now.")

        n_ics = train_logits.shape[1]
        n_thresholds = 500
        thresholds = torch.tensor([x / n_thresholds for x in range(n_thresholds + 1)])
        costs = data["ee_eval"][n_tasks - 1]["exit_costs"]
        baseline_cost = data["ee_eval"][n_tasks - 1]["baseline_cost"]
        costs = torch.tensor(costs) / baseline_cost

        train_logits = train_logits.to(args.device, non_blocking=True)
        test_logits = test_logits.to(args.device, non_blocking=True)
        test_targets = test_targets.to(args.device, non_blocking=True)
        train_targets = train_targets.to(args.device, non_blocking=True)
        costs = costs.to(args.device, non_blocking=True)
        thresholds = thresholds.to(args.device, non_blocking=True)

        plot_data = []
        metrics = []

        bias_data_path = output_path / "data.pt"
        heatmap_path = output_path / "bias.png"

        if bias_data_path.exists():
            best_bias = torch.load(bias_data_path)
            print(
                f"\tFound computed TLC data at {bias_data_path}, skipping the optimization"
            )
        else:
            print("\tOptimizing TLC...")
            best_bias, loss_curve = optimize_tlc(
                logits=train_logits,
                classes_per_task=classes_per_task,
                optimization_algorithm=args.optimization_algorithm,
                max_iters=args.max_iters,
                hp_space=args.hp_space,
                hp_mu=args.hp_mu,
                hp_sigma=args.hp_sigma,
                hp_min=args.hp_min,
                hp_max=args.hp_max,
            )
            torch.save(best_bias, bias_data_path)
            loss_path = output_path / "loss.png"
            plot_loss(loss_curve, loss_path)
            plot_heatmap(best_bias, heatmap_path)

        best_bias = expand_bias(best_bias, classes_per_task)
        adapted_logits = test_logits + best_bias.unsqueeze(0)
        adapted_probs = torch.nn.functional.softmax(adapted_logits, dim=-1)
        ee_acc, ee_cost = evaluate_ee(
            adapted_probs,
            test_targets,
            costs,
            thresholds,
            return_max_conf_on_no_exit=True,
        )
        for acc, cost in zip(ee_acc.tolist(), ee_cost.tolist()):
            plot_data.append({"method": "tlc", "acc": acc, "cost": cost})

        ee_metrics = compute_metrics(ee_acc, ee_cost)
        metrics.append({"method": "tlc", **ee_metrics})

        base_acc, base_cost = evaluate_ee(
            probs=torch.nn.functional.softmax(test_logits, dim=-1),
            targets=test_targets,
            exit_costs=costs,
            thresholds=thresholds,
        )
        for acc, cost in zip(base_acc.tolist(), base_cost.tolist()):
            plot_data.append({"method": "base", "acc": acc, "cost": cost})
        ee_metrics = compute_metrics(base_acc, base_cost)
        metrics.append({"method": "base", **ee_metrics})
        evaluate_ics(
            probs=torch.nn.functional.softmax(test_logits, dim=-1),
            targets=test_targets,
            n_tasks=n_tasks,
            output_path=output_path.joinpath("ic_stats.pt"),
        )

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_path / "metrics.csv", index=False)

        df = pd.DataFrame(plot_data)
        df.to_csv(output_path / "data.csv", index=False)
        plot_cost_vs_acc(df, output_path / "cost_vs_acc.png")
