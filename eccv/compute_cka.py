from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from metrics import _CKA

if __name__ == "__main__":
    # read all features from the root directory
    root = Path(__file__).parent
    feature_dirs = list(root.rglob("features"))
    device = 'cuda'

    with torch.no_grad():
        for feature_dir in tqdm(feature_dirs):
            # get the path to the features file and create matching output dir
            new_dir = root / 'cka' / (feature_dir.relative_to(root)).parent
            new_dir.mkdir(parents=True, exist_ok=True)

            # load the features for each task in features dir
            task_dirs = sorted(list(feature_dir.glob('test_*')))
            task_to_features = {}
            for task_dir in task_dirs:
                task_idx = int(task_dir.name.replace('test_', ''))
                task_files = sorted(list(task_dir.glob('*.pt')))
                task_ids = sorted(list(set([task_file.name.split('_')[1] for task_file in task_files])))

                per_task_features = {}
                for id in task_ids:
                    batch_files = [task_file for task_file in task_files if f'task_{id}' in task_file.name]
                    # load batches, list of tensors for each intermediate layer
                    batches = [torch.load(batch_file, map_location='cpu') for batch_file in batch_files]
                    # concatenate features for each intermediate layer
                    features = [torch.cat([batch[i] for batch in batches], dim=0) for i in range(len(batches[0]))]
                    per_task_features[int(id)] = features
                task_to_features[int(task_idx)] = per_task_features
            n_tasks = len(task_to_features)
            ckas = torch.zeros((7, n_tasks, n_tasks))
            for task_id in range(n_tasks):
                task_data = [task_to_features[i][task_id] for i in range(task_id, n_tasks)]
                task_data = [None] * (n_tasks - len(task_data)) + task_data
                for feat_idx in range(7):
                    for j in range(len(task_data) - 1):
                        if task_data[j] is None:
                            continue
                        feats_cur_task = task_data[j][feat_idx]
                        feats_next_task = task_data[j + 1][feat_idx]
                        feats_cur_task = feats_cur_task.view(feats_cur_task.shape[0], -1)
                        feats_next_task = feats_next_task.view(feats_next_task.shape[0], -1)
                        ckas[feat_idx, task_id, j + 1] = _CKA(feats_cur_task.to(device), feats_next_task.to(device)).item()

            torch.save(ckas, new_dir / 'cka.pt')