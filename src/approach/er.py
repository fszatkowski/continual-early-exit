import time
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

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
        logger=None,
        exemplars_dataset=None,
        scheduler_milestones=None,
        all_outputs=False,
        no_learning=False,
    ):
        super(Appr, self).__init__(
            model,
            device,
            nepochs,
            lr,
            lr_min,
            lr_factor,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            multi_softmax,
            fix_bn,
            eval_on_train,
            select_best_model_by_val_loss,
            logger,
            exemplars_dataset,
            scheduler_milestones,
            no_learning,
        )
        self.all_out = all_outputs

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--all-outputs",
            action="store_true",
            required=False,
            help="Allow all weights related to all outputs to be modified (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if (
            len(self.exemplars_dataset) == 0
            and len(self.model.heads) > 1
            and not self.all_out
        ):
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(
                self.model.heads[-1].parameters()
            )
        else:
            params = self.model.parameters()
        return torch.optim.SGD(
            params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum
        )

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader, mem_loader=None)
        self.post_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader, mem_loader=None):
        """Contains the epochs loop"""

        # add exemplars to mem_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            mem_loader = torch.utils.data.DataLoader(
                self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

            total_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )
        else:
            total_loader = trn_loader

            # FINETUNING TRAINING -- contains the epochs loop
        self.train_loop(t, trn_loader, val_loader, mem_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        # import pdb; pdb.set_trace()
        self.exemplars_dataset.collect_exemplars(
            self.model, total_loader, val_loader.dataset.transform
        )

    def train_loop(self, t, trn_loader, val_loader, mem_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        self.optimizer = self._get_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.decay_milestones, gamma=self.lr_decay
        )

        max_coeffs = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9])

        # Loop epochs
        for e in range(self.nepochs):
            cur_coeffs = 0.01 + e * (max_coeffs / (self.nepochs - 1))
            cur_coeffs = np.minimum(max_coeffs, cur_coeffs)
            print("\nCur coeffs: {}".format(cur_coeffs))

            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, mem_loader, cur_coeffs)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader)
                clock2 = time.time()
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
                for ic_id, loss in enumerate(train_loss.tolist()):
                    self.logger.log_scalar(
                        task=None,
                        iter=None,
                        name=f"ic_{ic_id}",
                        value=loss,
                        group="train_loss",
                    )
                for ic_id, acc in enumerate(train_acc.tolist()):
                    self.logger.log_scalar(
                        task=None,
                        iter=None,
                        name=f"ic_{ic_id}",
                        value=100 * acc,
                        group="train_acc",
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
            print(
                " Valid: time={:5.1f}s loss={:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f} TAw acc={:5.1f}%,{:5.1f}%,{:5.1f}%,{:5.1f}%,{:5.1f}%,{:5.1f}%,{:5.1f}% |".format(
                    clock4 - clock3,
                    valid_loss[0],
                    valid_loss[1],
                    valid_loss[2],
                    valid_loss[3],
                    valid_loss[4],
                    valid_loss[5],
                    valid_loss[6],
                    100 * valid_acc[0],
                    100 * valid_acc[1],
                    100 * valid_acc[2],
                    100 * valid_acc[3],
                    100 * valid_acc[4],
                    100 * valid_acc[5],
                    100 * valid_acc[6],
                ),
                end="",
            )
            for ic_id, loss in enumerate(valid_loss.tolist()):
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name=f"ic_{ic_id}",
                    value=loss,
                    group="valid_loss",
                )
            for ic_id, acc in enumerate(valid_acc.tolist()):
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name=f"ic_{ic_id}",
                    value=100 * acc,
                    group="valid_acc",
                )

            scheduler.step()
            print(" lr={:.1e}".format(self.optimizer.param_groups[0]["lr"]), end="")
            self.logger.log_scalar(
                task=t, iter=e + 1, name="lr", value=lr, group="train"
            )
            print()

    def train_epoch(self, t, trn_loader, mem_loader, cur_coeffs):
        """Runs a single epoch"""
        self.model.train()
        # if self.fix_bn and t > 0:
        # self.model.freeze_bn()
        if t > 0 and mem_loader is not None:
            mem_iter = iter(mem_loader)
        for images, targets in trn_loader:
            if t > 0 and mem_loader is not None:
                try:
                    images_mem, targets_mem = next(mem_iter)
                except StopIteration:
                    mem_iter = iter(mem_loader)
                    images_mem, targets_mem = next(mem_iter)
                # images_mem, targets_mem = mem_loader[0]
                images = torch.cat((images, images_mem))
                targets = torch.cat((targets, targets_mem))

            total_loss = 0.0
            # Forward current model
            outputs = self.model(images.to(self.device))
            for ic_id, output in enumerate(outputs[:6]):
                total_loss += float(cur_coeffs[ic_id]) * self.criterion(
                    t, output, targets.to(self.device)
                )
            total_loss += self.criterion(t, outputs[-1], targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        return torch.nn.functional.cross_entropy(
            outputs[t], targets - self.model.task_offset[t]
        )
