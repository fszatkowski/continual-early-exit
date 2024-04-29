import logging
import time
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from datasets.exemplars_dataset import ExemplarsDataset
from early_exits.flops import analyze_flops
from loggers.exp_logger import ExperimentLogger


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(
        self,
        model,
        device,
        nepochs=100,
        lr=0.05,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0,
        wd=0,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger: ExperimentLogger = None,
        exemplars_dataset: ExemplarsDataset = None,
        scheduler_milestones=None,
        no_learning=False,
    ):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.select_best_model_by_val_loss = select_best_model_by_val_loss
        self.optimizer = None
        self.scheduler_milestones = scheduler_milestones
        self.scheduler = None
        self.debug = False
        self.no_learning = no_learning
        self.current_epoch = 0

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0:
            # No exemplars case
            base_params = list(self.model.model.parameters())
            if self.model.is_early_exit():
                head_params = [
                    p
                    for ic_heads in self.model.heads
                    for p in ic_heads[-1].parameters()
                ]
            else:
                head_params = list(self.model.heads[-1].parameters())
            params = base_params + head_params
        else:
            params = list(self.model.parameters())
        return torch.optim.SGD(
            params,
            lr=self.lr,
            weight_decay=self.wd,
            momentum=self.momentum,
        )

    def _get_scheduler(self):
        if self.scheduler_milestones is not None:
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.scheduler_milestones,
                gamma=0.1,
            )
        else:
            return None

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        pass

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.current_epoch = e
            self.train_epoch(t, trn_loader)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
                self.log_results(
                    t=t, e=e, loss=train_loss, acc=train_acc, group="train"
                )
                if self.model.is_early_exit():
                    train_loss = train_loss[-1]
                    train_acc = train_acc[-1]
                print(
                    "| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |".format(
                        e + 1,
                        clock1 - clock0,
                        clock2 - clock1,
                        train_loss,
                        100 * train_acc,
                    ),
                    end="",
                )
            else:
                print(
                    "| Epoch {:3d}, time={:5.1f}s | Train: skip eval |".format(
                        e + 1, clock1 - clock0
                    ),
                    end="",
                )

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader)
            clock4 = time.time()
            self.log_results(t=t, e=e, loss=valid_loss, acc=valid_acc, group="valid")
            if self.model.is_early_exit():
                valid_loss = valid_loss[-1]
                valid_acc = valid_acc[-1]
            print(
                " Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |".format(
                    clock4 - clock3, valid_loss, 100 * valid_acc
                ),
                end="",
            )

            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()[0]

            if self.select_best_model_by_val_loss:
                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    best_model = self.model.get_copy()
                    patience = self.lr_patience
                    print(" *", end="")
                else:
                    patience -= 1
                    if self.scheduler is None:
                        # if the loss does not go down, decrease patience
                        if patience <= 0:
                            # if it runs out of patience, reduce the learning rate
                            lr /= self.lr_factor
                            print(" lr={:.1e}".format(lr), end="")
                            if lr < self.lr_min:
                                # if the lr decreases below minimum, stop the training session
                                print()
                                break
                            # reset patience and recover best model so far to continue training
                            patience = self.lr_patience
                            self.optimizer.param_groups[0]["lr"] = lr
                            self.model.set_state_dict(best_model)
            else:
                best_model = self.model.get_copy()

            self.logger.log_scalar(
                task=t, iter=e + 1, name="patience", value=patience, group="train"
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="lr", value=lr, group="train"
            )
            print()

        self.model.set_state_dict(best_model)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward current model
            outputs = self.model(images)
            loss = self.criterion(t, outputs, targets)
            if self.model.is_early_exit():
                loss = sum(loss)
            # Backward
            if t == 0 or not self.no_learning:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            if self.model.is_early_exit():
                total_loss, total_acc_taw, total_acc_tag, total_num = (
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                    np.zeros((len(self.model.ic_layers) + 1,)),
                )
            else:
                total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                images, targets = images.to(self.device, non_blocking=True), targets.to(
                    self.device, non_blocking=True
                )

                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets)

                if self.model.is_early_exit():
                    for i, ic_outputs in enumerate(outputs):
                        hits_taw, hits_tag = self.calculate_metrics(ic_outputs, targets)
                        # Log
                        total_loss += loss[i].item() * len(targets)
                        total_acc_taw[i] += hits_taw.sum().item()
                        total_acc_tag[i] += hits_tag.sum().item()
                        total_num += len(targets)
                else:
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    # Log
                    total_loss += loss.item() * len(targets)
                    total_acc_taw += hits_taw.sum().item()
                    total_acc_tag += hits_tag.sum().item()
                    total_num += len(targets)

            if self.model.is_early_exit():
                total_num //= len(self.model.ic_layers) + 1

        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def ee_net(self, exit_layer: Optional[int] = None) -> torch.nn.Module:
        # early exit network with all CL logic implemented for inference, for profiling
        tmp_model = deepcopy(self.model)
        tmp_model.set_exit_layer(exit_layer)
        return tmp_model

    def eval_early_exit(
        self,
        t,
        dataloader,
        thresholds,
        exit_costs=None,
        baseline_cost=None,
        no_exit_strategy="max_conf",
        subset="test",
    ):
        """Evaluate early exit properties of the network"""
        """ Evaluate per-IC cost of the inference """
        self.model.eval()
        with torch.no_grad():
            if exit_costs is None:
                logging.info("Profiling FLOPs per-layer...")
                samples, _ = next(iter(deepcopy(dataloader)))
                sample = samples[:1].to(self.device)
                exit_costs = []
                for exit_idx in range(len(self.model.ic_layers) + 1):
                    net_for_profiling = self.ee_net(exit_idx)
                    flops, params = analyze_flops(net_for_profiling, sample)
                    exit_costs.append(flops)
                exit_costs = np.array(exit_costs)

                net_for_profiling = self.ee_net(-1)
                flops, params = analyze_flops(net_for_profiling, sample)
                baseline_cost = flops

            n_cls = len(self.model.ic_layers) + 1

            inference_network = self.ee_net()
            thresholds = (
                thresholds.unsqueeze(1).unsqueeze(1).to(self.device)
            )  # [th, batch size, ic idx]
            per_ic_hits = {
                "taw": np.zeros(
                    n_cls,
                ),
                "tag": np.zeros(
                    n_cls,
                ),
            }
            per_th_hits = {
                "taw": np.zeros((len(thresholds),)),
                "tag": np.zeros((len(thresholds),)),
            }
            per_th_exits = {
                "taw": np.zeros((len(thresholds), n_cls)),
                "tag": np.zeros((len(thresholds), n_cls)),
            }

            total_cnt = 0
            batch_idx = 0
            for images, targets in dataloader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = inference_network(images)

                merged_outputs_taw = torch.stack(
                    [o[t] for o in outputs], dim=1
                )  # [batch size, ic idx, num classes]
                merged_outputs_tag = torch.stack(
                    [
                        torch.cat([torch.cat(ic_outputs, dim=1)], dim=-1)
                        for ic_outputs in outputs
                    ],
                    dim=1,
                )  # [batch size, ic idx, num classes]

                preds_taw_static = (
                    merged_outputs_taw.argmax(dim=-1) + self.model.task_offset[t]
                )  # [batch size, ic idx]
                preds_tag_static = merged_outputs_tag.argmax(
                    dim=-1
                )  # [batch size, ic idx]

                hits_taw_static = (preds_taw_static == targets.unsqueeze(-1)).sum(
                    dim=0
                )  # [ic idx]
                hits_tag_static = (preds_tag_static == targets.unsqueeze(-1)).sum(
                    dim=0
                )  # [ic idx]

                per_ic_hits["taw"] += hits_taw_static.cpu().numpy()
                per_ic_hits["tag"] += hits_tag_static.cpu().numpy()

                merged_probs_taw = merged_outputs_taw.softmax(dim=-1).unsqueeze(
                    0
                )  # [th, batch size, ic idx, num classes]
                merged_probs_tag = merged_outputs_tag.softmax(dim=-1).unsqueeze(
                    0
                )  # [th, batch size, ic idx, num classes]

                merged_max_conf_taw = merged_probs_taw.max(dim=-1)[0]
                merged_max_conf_tag = merged_probs_tag.max(dim=-1)[0]

                th_mask_taw = merged_max_conf_taw >= thresholds
                th_mask_tag = merged_max_conf_tag >= thresholds

                th_mask_taw = th_mask_taw.float()
                th_mask_tag = th_mask_tag.float()

                # Compute mask determining if early exit happened
                exit_mask_taw = th_mask_taw.sum(dim=-1) > 0
                exit_mask_tag = th_mask_tag.sum(dim=-1) > 0

                # Compute exit indices
                exit_ic_idx_taw = th_mask_taw.argmax(dim=-1)
                exit_ic_idx_tag = th_mask_tag.argmax(dim=-1)

                # Choose predictions for samples which didn't exit
                if no_exit_strategy == "max_conf":
                    no_exit_ic_idx_taw = merged_max_conf_taw.argmax(dim=2)
                    no_exit_ic_idx_tag = merged_max_conf_tag.argmax(dim=2)
                elif no_exit_strategy == "last":
                    no_exit_ic_idx_taw = n_cls - 1
                    no_exit_ic_idx_tag = n_cls - 1
                else:
                    raise NotImplementedError()

                # Compute real exit layer for each sample (some samples might not exit and chose previous layer, but their cost is full net)
                real_exit_idx_taw = torch.where(
                    exit_mask_taw, exit_ic_idx_taw, n_cls - 1
                )
                real_exit_idx_tag = torch.where(
                    exit_mask_tag, exit_ic_idx_tag, n_cls - 1
                )

                # Compute per threshold predictions
                pred_idx_taw = torch.where(
                    exit_mask_taw, exit_ic_idx_taw, no_exit_ic_idx_taw
                )
                pred_idx_tag = torch.where(
                    exit_mask_tag, exit_ic_idx_tag, no_exit_ic_idx_tag
                )

                preds_mask_taw = torch.nn.functional.one_hot(
                    pred_idx_taw, num_classes=n_cls
                )
                preds_mask_tag = torch.nn.functional.one_hot(
                    pred_idx_tag, num_classes=n_cls
                )

                preds_taw_dynamic = (
                    preds_taw_static.unsqueeze(0) * preds_mask_taw
                ).sum(dim=-1) + self.model.task_offset[t]
                preds_tag_dynamic = (
                    preds_tag_static.unsqueeze(0) * preds_mask_tag
                ).sum(dim=-1)

                hits_taw_dynamic = (preds_taw_dynamic == targets.unsqueeze(0)).sum(
                    dim=1
                )
                hits_tag_dynamic = (preds_tag_dynamic == targets.unsqueeze(0)).sum(
                    dim=1
                )

                per_th_hits["taw"] += hits_taw_dynamic.cpu().numpy()
                per_th_hits["tag"] += hits_tag_dynamic.cpu().numpy()

                per_th_exits["taw"] += (
                    torch.nn.functional.one_hot(real_exit_idx_taw, num_classes=n_cls)
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )
                per_th_exits["tag"] += (
                    torch.nn.functional.one_hot(real_exit_idx_tag, num_classes=n_cls)
                    .sum(dim=1)
                    .cpu()
                    .numpy()
                )

                total_cnt += len(targets)
                output_dir = Path(self.logger.exp_path) / f"logits_{subset}" / f"t_{t}"
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "outputs_tag": merged_outputs_tag,
                        "outputs_taw": merged_outputs_taw,
                        "targets": targets,
                    },
                    output_dir / f"{batch_idx}.pt",
                )
                batch_idx += 1

            per_ic_accuracy = {
                acc_type: hits / total_cnt for acc_type, hits in per_ic_hits.items()
            }

            per_th_accuracy = {
                acc_type: hits / total_cnt for acc_type, hits in per_th_hits.items()
            }
            per_th_exit_cnt = {
                acc_type: cnt / total_cnt for acc_type, cnt in per_th_exits.items()
            }
            return (
                exit_costs,
                baseline_cost,
                per_ic_accuracy,
                per_th_accuracy,
                per_th_exit_cnt,
            )

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        pred = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (
                self.model.task_cls.cumsum(0).to(self.device, non_blocking=True)
                <= targets[m]
            ).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets).float()
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [
                torch.nn.functional.log_softmax(output, dim=1) for output in outputs
            ]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets).float()
        return hits_taw, hits_tag

    def log_results(self, t, e, loss, acc, group):
        if self.model.is_early_exit():
            for i, (l, a) in enumerate(zip(loss[:-1], acc[:-1])):
                self.logger.log_scalar(
                    task=t, iter=e + 1, name=f"loss_c_{i}", value=l, group=group
                )
                self.logger.log_scalar(
                    task=t, iter=e + 1, name=f"acc_c_{i}", value=100 * a, group=group
                )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="loss", value=loss[-1], group=group
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="acc", value=100 * acc[-1], group=group
            )
        else:
            self.logger.log_scalar(
                task=t, iter=e + 1, name="loss", value=loss, group=group
            )
            self.logger.log_scalar(
                task=t, iter=e + 1, name="acc", value=100 * acc, group=group
            )

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            loss = []
            for ic_outputs, ic_weight in zip(outputs, ic_weights):
                loss.append(
                    ic_weight
                    * torch.nn.functional.cross_entropy(
                        ic_outputs[t], targets - self.model.task_offset[t]
                    )
                )
            return loss
        else:
            return torch.nn.functional.cross_entropy(
                outputs[t], targets - self.model.task_offset[t]
            )
