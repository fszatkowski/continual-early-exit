import itertools
from argparse import ArgumentParser

import torch

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Elastic Weight Consolidation (EWC) approach
    described in http://arxiv.org/abs/1612.00796
    """

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
        lamb=5000,
        alpha=0.5,
        fi_sampling_type="max_pred",
        fi_num_samples=-1,
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
        )
        self.lamb = lamb
        self.alpha = alpha
        self.sampling_type = fi_sampling_type
        self.num_samples = fi_num_samples

        feat_ext = self.model.model
        # Store current parameters as the initial parameters before first task starts
        older_params = {
            n: p.clone().detach()
            for n, p in feat_ext.named_parameters()
            if p.requires_grad
        }
        # Store fisher information weight importance
        fisher = {
            n: torch.zeros(p.shape).to(self.device, non_blocking=True)
            for n, p in feat_ext.named_parameters()
            if p.requires_grad
        }

        if self.model.is_early_exit():
            self.older_params = []
            self.fisher = []
            self.fisher_modules = []

            for ic_layer_name in self.model.ic_layers:
                feat_ext = self._get_module_by_name(ic_layer_name)
                ic_older_params = {
                    n: p.clone().detach()
                    for n, p in feat_ext.named_parameters()
                    if p.requires_grad
                }
                # Store fisher information weight importance
                ic_fisher = {
                    n: torch.zeros(p.shape).to(self.device, non_blocking=True)
                    for n, p in feat_ext.named_parameters()
                    if p.requires_grad
                }
                self.older_params.append(ic_older_params)
                self.fisher.append(ic_fisher)
                self.fisher_modules.append(feat_ext)
            self.older_params.append(older_params)
            self.fisher.append(fisher)
            self.fisher_modules.append(self.model.model)
        else:
            self.older_params = older_params
            self.fisher = fisher

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Eq. 3: "lambda sets how important the old task is compared to the new one"
        parser.add_argument(
            "--lamb",
            default=5000,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        # Define how old and new fisher is fused, by default it is a 50-50 fusion
        parser.add_argument(
            "--alpha",
            default=0.5,
            type=float,
            required=False,
            help="EWC alpha (default=%(default)s)",
        )
        parser.add_argument(
            "--fi-sampling-type",
            default="max_pred",
            type=str,
            required=False,
            choices=["true", "max_pred", "multinomial"],
            help="Sampling type for Fisher information (default=%(default)s)",
        )
        parser.add_argument(
            "--fi-num-samples",
            default=-1,
            type=int,
            required=False,
            help="Number of samples for Fisher information (-1: all available) (default=%(default)s)",
        )

        return parser.parse_known_args(args)

    def _get_module_by_name(self, module_name):
        feat_ext = None
        for name, module in self.model.model.named_modules():
            if name == module_name:
                return module
        if feat_ext is None:
            raise ValueError(f"Module {module_name} not found")

    def compute_fisher_matrix_diag(self, trn_loader):
        # Store Fisher Information
        if self.model.is_early_exit():
            fisher = []
            for ic_layer_name in self.model.ic_layers:
                feat_ext = self._get_module_by_name(ic_layer_name)
                # Store fisher information weight importance
                ic_fisher = {
                    n: torch.zeros(p.shape).to(self.device, non_blocking=True)
                    for n, p in feat_ext.named_parameters()
                    if p.requires_grad
                }
                fisher.append(ic_fisher)
            fisher.append(
                {
                    n: torch.zeros(p.shape).to(self.device, non_blocking=True)
                    for n, p in self.model.model.named_parameters()
                    if p.requires_grad
                }
            )
        else:
            fisher = {
                n: torch.zeros(p.shape).to(self.device, non_blocking=True)
                for n, p in self.model.model.named_parameters()
                if p.requires_grad
            }
        # Compute fisher information for specified number of samples -- rounded to the batch size
        n_samples_batches = (
            (self.num_samples // trn_loader.batch_size + 1)
            if self.num_samples > 0
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        )
        # Do forward and backward pass to compute the fisher information
        self.model.train()
        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            outputs = self.model.forward(images)

            # Compute fisher information
            if self.model.is_early_exit():
                for ic_idx in range(len(outputs)):
                    ic_outputs = outputs[ic_idx]
                    ic_module = self.fisher_modules[ic_idx]
                    if self.sampling_type == "true":
                        # Use the labels to compute the gradients based on the CE-loss with the ground truth
                        preds = targets
                    elif self.sampling_type == "max_pred":
                        # Not use labels and compute the gradients related to the prediction the model has learned
                        preds = torch.cat(ic_outputs, dim=1).argmax(1).flatten()
                    elif self.sampling_type == "multinomial":
                        # Use a multinomial sampling to compute the gradients
                        probs = torch.nn.functional.softmax(
                            torch.cat(ic_outputs, dim=1), dim=1
                        )
                        preds = torch.multinomial(probs, len(targets)).flatten()
                    loss = torch.nn.functional.cross_entropy(
                        torch.cat(ic_outputs, dim=1), preds
                    )
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    # Accumulate all gradients from loss with regularization
                    for n, p in ic_module.named_parameters():
                        if p.grad is not None:
                            fisher[ic_idx][n] += p.grad.pow(2) * len(targets)
            else:
                if self.sampling_type == "true":
                    # Use the labels to compute the gradients based on the CE-loss with the ground truth
                    preds = targets
                elif self.sampling_type == "max_pred":
                    # Not use labels and compute the gradients related to the prediction the model has learned
                    preds = torch.cat(outputs, dim=1).argmax(1).flatten()
                elif self.sampling_type == "multinomial":
                    # Use a multinomial sampling to compute the gradients
                    probs = torch.nn.functional.softmax(
                        torch.cat(outputs, dim=1), dim=1
                    )
                    preds = torch.multinomial(probs, len(targets)).flatten()

                loss = torch.nn.functional.cross_entropy(
                    torch.cat(outputs, dim=1), preds
                )
                self.optimizer.zero_grad()
                loss.backward()
                # Accumulate all gradients from loss with regularization
                for n, p in self.model.model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.pow(2) * len(targets)
            # Apply mean across all samples
            n_samples = n_samples_batches * trn_loader.batch_size
            if self.model.is_early_exit():
                for ic_idx in range(len(fisher)):
                    fisher[ic_idx] = {
                        n: (p / n_samples) for n, p in fisher[ic_idx].items()
                    }
            else:
                fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(
                trn_loader.dataset + self.exemplars_dataset,
                batch_size=trn_loader.batch_size,
                shuffle=True,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Store current parameters for the next task
        if self.model.is_early_exit():
            for ic_idx in range(len(self.older_params)):
                self.older_params[ic_idx] = {
                    n: p.clone().detach()
                    for n, p in self.fisher_modules[ic_idx].named_parameters()
                    if p.requires_grad
                }
        else:
            self.older_params = {
                n: p.clone().detach()
                for n, p in self.model.model.named_parameters()
                if p.requires_grad
            }

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(trn_loader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        if self.model.is_early_exit():
            for ic_idx, ic_fisher in enumerate(self.fisher):
                curr_ic_fisher = curr_fisher[ic_idx]
                for n in ic_fisher.keys():
                    if self.alpha == -1:
                        alpha = (
                            sum(self.model.task_cls[:t]) / sum(self.model.task_cls)
                        ).to(self.device, non_blocking=True)
                        ic_fisher[n] = (
                            alpha * ic_fisher[n] + (1 - alpha) * curr_ic_fisher[n]
                        )
                    else:
                        ic_fisher[n] = (
                            self.alpha * ic_fisher[n]
                            + (1 - self.alpha) * curr_ic_fisher[n]
                        )
        else:
            for n in self.fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing alpha
                if self.alpha == -1:
                    alpha = (
                        sum(self.model.task_cls[:t]) / sum(self.model.task_cls)
                    ).to(self.device, non_blocking=True)
                    self.fisher[n] = (
                        alpha * self.fisher[n] + (1 - alpha) * curr_fisher[n]
                    )
                else:
                    self.fisher[n] = (
                        self.alpha * self.fisher[n] + (1 - self.alpha) * curr_fisher[n]
                    )

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.model.is_early_exit():
            loss = []
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            for ic_idx in range(len(outputs)):
                ic_outputs = outputs[ic_idx]
                ic_loss_reg = 0

                if t > 0:
                    ic_fisher = self.fisher[ic_idx]
                    ic_older_params = self.older_params[ic_idx]
                    ic_fisher_module = self.fisher_modules[ic_idx]
                    for n, p in ic_fisher_module.named_parameters():
                        if n in ic_fisher.keys():
                            ic_loss_reg += (
                                torch.sum(
                                    ic_fisher[n] * (p - ic_older_params[n]).pow(2)
                                )
                                / 2
                            )

                # Current cross-entropy loss -- with exemplars use all heads
                if len(self.exemplars_dataset) > 0:
                    ic_loss_ce = torch.nn.functional.cross_entropy(
                        torch.cat(ic_outputs, dim=1), targets
                    )
                else:
                    ic_loss_ce = torch.nn.functional.cross_entropy(
                        ic_outputs[t], targets - self.model.task_offset[t]
                    )
                ic_loss = ic_loss_ce + self.lamb * ic_loss_reg
                ic_loss = ic_weights[ic_idx] * ic_loss
                loss.append(ic_loss)
        else:
            loss_reg = 0
            if t > 0:
                # Eq. 3: elastic weight consolidation quadratic penalty
                for n, p in self.model.model.named_parameters():
                    if n in self.fisher.keys():
                        loss_reg += (
                            torch.sum(
                                self.fisher[n] * (p - self.older_params[n]).pow(2)
                            )
                            / 2
                        )
            # Current cross-entropy loss -- with exemplars use all heads
            if len(self.exemplars_dataset) > 0:
                loss_ce = torch.nn.functional.cross_entropy(
                    torch.cat(outputs, dim=1), targets
                )
            else:
                loss_ce = torch.nn.functional.cross_entropy(
                    outputs[t], targets - self.model.task_offset[t]
                )
            loss = loss_ce + self.lamb * loss_reg
        return loss
