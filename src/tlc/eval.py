from pathlib import Path

import torch
from torch import Tensor


def auc(accs: Tensor, costs: Tensor):
    # accs, costs: tensors of shape [N,]
    avg_accs = (accs[:-1] + accs[1:]) / 2
    costs = costs[1:] - costs[:-1]
    return (avg_accs * costs).sum()


def get_acc_for_budget(accs, costs, budget):
    accs, costs = accs.tolist(), costs.tolist()
    under_budget_vals = [c for c in costs if c <= budget]
    if len(under_budget_vals) == 0:
        return -1
    budget_val = max(under_budget_vals)
    budget_idx = costs.index(budget_val)
    budget_acc = accs[budget_idx]
    return budget_acc


def compute_metrics(accs: Tensor, costs: Tensor):
    output = {"auc": float(auc(accs, costs))}
    for budget in [0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0]:
        output[f"acc_{budget}"] = get_acc_for_budget(accs, costs, budget)
    return output


def evaluate_ee(
    probs: Tensor,
    targets: Tensor,
    exit_costs: Tensor,
    thresholds: Tensor,
    return_max_conf_on_no_exit=True,
):
    # probs: tensor of shape [batch_size, n_ics, n_classes]
    # thresholds: tensor of shape [N_thresholds]
    # exit_costs: tensor of shape [n_ics]

    _, n_ics, n_classes = probs.shape
    max_ic_val = torch.tensor(n_ics - 1).to(probs.device)

    max_confidence_values, max_confidence_preds = torch.max(probs, dim=2)
    max_confidence_values = max_confidence_values.unsqueeze(
        dim=0
    )  # shape [1, batch_size, n_ics]
    max_confidence_preds = max_confidence_preds.unsqueeze(
        dim=0
    )  # shape [1, batch_size, n_ics]
    thresholds_tensor = thresholds.unsqueeze(dim=-1).unsqueeze(
        dim=-1
    )  # shape [N_thresholds, 1, 1]
    exit_costs_tensor = exit_costs.unsqueeze(dim=0).unsqueeze(
        dim=0
    )  # shape [1, 1, n_ics]

    # Compute exit masks
    th_satisfied = (
        max_confidence_values >= thresholds_tensor
    )  # shape [N_thresholds, batch_size, n_ics]
    exited = th_satisfied.sum(dim=2)  # shape [N_thresholds, batch_size]
    min_exit_idx = th_satisfied.float().argmax(
        dim=2
    )  # shape [N_thresholds, batch_size]

    # Set exit id to the exit idx if network exited, else return max IC idx
    exit_idx = torch.where(
        exited != 0, min_exit_idx, max_ic_val
    )  # shape [N_thresholds, batch_size]

    # Compute predictions for each sample
    # Get predictions for samples that exited
    per_ic_preds_exited = (
        torch.nn.functional.one_hot(min_exit_idx, num_classes=n_ics)
        * max_confidence_preds
    ).sum(
        dim=2
    )  # shape [N_thresholds, batch_size]

    if return_max_conf_on_no_exit:
        # Get predictions for samples that didn't exit
        max_conf_pred_values = torch.argmax(
            max_confidence_values, dim=2
        )  # shape [1, batch_size]
        max_conf_preds_mask = torch.nn.functional.one_hot(
            max_conf_pred_values, num_classes=n_ics
        )  # shape [1, batch_size, n_ics]
        max_conf_preds = (max_conf_preds_mask * max_confidence_preds).sum(
            dim=2
        )  # shape [1, batch_size]
        exit_output = torch.where(
            exited != 0, per_ic_preds_exited, max_conf_preds
        )  # shape [N_thresholds, batch_size]
    else:
        last_layer_preds = max_confidence_preds[:, :, -1]  # shape [1, batch_size]
        exit_output = torch.where(exited != 0, per_ic_preds_exited, last_layer_preds)

    # Compute cost for each sample
    exit_idx_mask = torch.nn.functional.one_hot(
        exit_idx, num_classes=n_ics
    )  # shape [N_thresholds, batch_size, n_ics]
    exit_costs_per_sample = (exit_costs_tensor * exit_idx_mask).sum(
        dim=2
    )  # shape [N_thresholds, batch_size, n_ics]

    targets_tensor = targets.unsqueeze(dim=0)  # shape [1, batch_size]
    acc_per_th = (exit_output == targets_tensor).sum(dim=1) / targets_tensor.shape[
        1
    ]  # shape [N_thresholds]
    cost_per_th = exit_costs_per_sample.mean(dim=1)  # shape [N_thresholds]

    return acc_per_th, cost_per_th


def evaluate_ics(probs: Tensor, targets: Tensor, n_tasks: int, output_path: Path):
    _, n_ics, n_classes = probs.shape

    per_task_data = []
    task_size = targets.shape[0] // n_tasks
    for i in range(n_tasks):
        targets_task = targets[i * task_size : (i + 1) * task_size].unsqueeze(1)
        preds_task = probs[i * task_size : (i + 1) * task_size].argmax(dim=-1)
        acc_task = (preds_task == targets_task).float().sum(dim=0) / targets_task.shape[
            0
        ]
        per_task_data.append(acc_task)

    accs = torch.stack(per_task_data, dim=0)
    torch.save(accs, output_path)
