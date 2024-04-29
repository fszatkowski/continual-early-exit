from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor


def load_dir(dir_path: Path) -> Tuple[Tensor, Tensor]:
    outputs_tag = []
    targets = []
    for p in dir_path.glob("*.pt"):
        data = torch.load(p, map_location="cpu")
        outputs_tag.append(data["outputs_tag"])
        targets.append(data["targets"])
    return torch.cat(outputs_tag, dim=0), torch.cat(targets, dim=0)


def load_data(train_glob: str, test_glob: str) -> Dict:
    # Shapes: outputs: [batch_size, n_ics, n_tasks], targets: [batch_size]
    data_dir = Path(".")
    data_paths_train = sorted(list(data_dir.glob(train_glob)))
    assert len(data_paths_train) > 0, f"No data found in {data_dir}"

    root_dirs = [path.parent for path in data_paths_train]
    root_to_data = {}
    for root in root_dirs:
        root_train_path = root / "logits_train"
        root_test_path = root / "logits_test"
        train_dir = list(root_train_path.glob("*"))[0]
        train_logits_tag, train_targets = load_dir(train_dir)
        n_tasks = int(train_dir.name.split("_")[1]) + 1
        output = {
            "n_tasks": n_tasks,
            "train_logits_tag": train_logits_tag,
            "train_targets": train_targets,
            "test_data_per_task": [],
        }
        test_dirs = list(root_test_path.glob("*"))
        for test_dir in test_dirs:
            test_logits_tag, test_targets = load_dir(test_dir)
            task_id = test_dir.name.split("_")[1]
            output["test_data_per_task"].append(
                {
                    "task_id": task_id,
                    "test_logits_tag": test_logits_tag,
                    "test_targets": test_targets,
                    "classes_per_task": int(len(test_targets.unique())),
                }
            )
        output["test_logits_tag"] = torch.cat(
            [o["test_logits_tag"] for o in output["test_data_per_task"]], dim=0
        )
        output["test_targets"] = torch.cat(
            [o["test_targets"] for o in output["test_data_per_task"]], dim=0
        )
        output["ee_eval"] = np.load(
            root / "results" / "ee_eval.npy", allow_pickle=True
        ).item()
        root_to_data[str(root)] = output

    return root_to_data
