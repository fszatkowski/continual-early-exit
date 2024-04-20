import warnings
from argparse import ArgumentParser
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.exemplars_dataset import ExemplarsDataset
from datasets.exemplars_selection import override_dataset_transform

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Incremental Classifier and Representation Learning (iCaRL) approach
    described in https://arxiv.org/abs/1611.07725
    Original code available at https://github.com/srebuffi/iCaRL
    """

    def __init__(
        self,
        model,
        device,
        nepochs=60,
        lr=0.5,
        lr_min=1e-4,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10000,
        momentum=0.9,
        wd=1e-5,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger=None,
        exemplars_dataset=None,
        scheduler_milestones=None,
        lamb=1,
        logit_conversion="inverse",
        ic_pooling="none",
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
        self.model_old = None
        self.lamb = lamb
        self.nmc_logit_conversion = logit_conversion
        self.ic_pooling = ic_pooling

        # iCaRL is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        if not have_exemplars:
            warnings.warn(
                "Warning: iCaRL is expected to use exemplars. Check documentation."
            )

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4. " allowing iCaRL to balance between CE and distillation loss."
        parser.add_argument(
            "--lamb",
            default=1,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        parser.add_argument(
            "--logit-conversion",
            default="inverse",
            type=str,
            choices=["inverse", "reverse", "pdf"],
            help="NMC distance to logits conversion (default=%(default)s)",
        )
        parser.add_argument(
            "--ic-pooling",
            default="none",
            type=str,
            choices=["none", "avg", "max", "combined"],
            help="Pooling strategy for intermediate features (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    # Algorithm 1: iCaRL NCM Classify
    def classify(self, task, features, means, targets):
        # expand means to all batch images
        means = torch.stack(means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features.view(features.shape[0], -1)
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (
            (features - means).pow(2).sum(1)
        )  # TODO removed squeeze - check if it's still ok?
        # Task-Aware Multi-Head
        num_cls = self.model.task_cls[task]
        offset = self.model.task_offset[task]
        pred = dists[:, offset : offset + num_cls].argmin(1)
        hits_taw = (pred + offset == targets.to(self.device, non_blocking=True)).float()
        # Task-Agnostic Multi-Head
        pred = dists.argmin(1)
        hits_tag = (pred == targets.to(self.device, non_blocking=True)).float()
        return hits_taw, hits_tag

    def compute_mean_of_exemplars(self, trn_loader, transform):
        # change transforms to evaluation for this calculation
        with override_dataset_transform(self.exemplars_dataset, transform) as _ds:
            # change dataloader so it can be fixed to go sequentially (shuffle=False), this allows to keep same order
            icarl_loader = DataLoader(
                _ds,
                batch_size=trn_loader.batch_size,
                shuffle=False,
                num_workers=trn_loader.num_workers,
                pin_memory=trn_loader.pin_memory,
            )
            # extract features from the model for all train samples
            # Page 2: "All feature vectors are L2-normalized, and the results of any operation on feature vectors,
            # e.g. averages are also re-normalized, which we do not write explicitly to avoid a cluttered notation."
            if self.model.is_early_exit():
                extracted_features = [[] for _ in range(len(self.model.ic_layers) + 1)]
            else:
                extracted_features = []
            extracted_targets = []
            with torch.no_grad():
                self.model.eval()
                for images, targets in icarl_loader:
                    _, feats = self.model(
                        images.to(self.device, non_blocking=True), return_features=True
                    )
                    # normalize
                    if self.model.is_early_exit():
                        for ic_idx in range(len(feats)):
                            ic_features = feats[ic_idx]
                            ic_features = pool_nmc_features(
                                ic_features, ic_idx, len(feats), self.ic_pooling
                            )
                            ic_features = ic_features.view(ic_features.shape[0], -1)
                            extracted_features[ic_idx].append(
                                ic_features / ic_features.norm(dim=1).view(-1, 1)
                            )
                    else:
                        extracted_features.append(feats / feats.norm(dim=1).view(-1, 1))
                    extracted_targets.extend(targets)

            if self.model.is_early_exit():
                extracted_features = [torch.cat(feats) for feats in extracted_features]
            else:
                extracted_features = torch.cat(extracted_features)
            extracted_targets = np.array(extracted_targets)
            for curr_cls in np.unique(extracted_targets):
                # get all indices from current class
                cls_ind = np.where(extracted_targets == curr_cls)[0]
                # get all extracted features for current class
                if self.model.is_early_exit():
                    for ic_idx in range(len(extracted_features)):
                        cls_feats = extracted_features[ic_idx][cls_ind]
                        # add the exemplars to the set and normalize
                        cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                        self.exemplar_means[ic_idx].append(cls_feats_mean)
                else:
                    cls_feats = extracted_features[cls_ind]
                    # add the exemplars to the set and normalize
                    cls_feats_mean = cls_feats.mean(0) / cls_feats.mean(0).norm()
                    self.exemplar_means.append(cls_feats_mean)

    # Algorithm 2: iCaRL Incremental Train
    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # remove mean of exemplars during training since Alg. 1 is not used during Alg. 2
        if self.model.is_early_exit():
            self.exemplar_means = [[] for _ in range(len(self.model.ic_layers) + 1)]
        else:
            self.exemplar_means = []

        # Algorithm 3: iCaRL Update Representation
        # Alg. 3. "form combined training set", add exemplars to train_loader
        if t > 0:
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
        # Algorithm 4: iCaRL ConstructExemplarSet and Algorithm 5: iCaRL ReduceExemplarSet
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

        # compute mean of exemplars
        self.compute_mean_of_exemplars(trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward old model
            outputs_old = None
            if t > 0:
                outputs_old = self.model_old(images.to(self.device, non_blocking=True))
            # Forward current model
            outputs = self.model(images.to(self.device, non_blocking=True))
            loss = self.criterion(
                t, outputs, targets.to(self.device, non_blocking=True), outputs_old
            )

            if self.model.is_early_exit():
                loss = sum(loss)
            assert not torch.isnan(loss), "Loss is NaN"

            # Backward
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
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                # Forward old model
                outputs_old = None
                if t > 0:
                    outputs_old = self.model_old(images)
                # Forward current model
                outputs, feats = self.model(
                    images.to(self.device, non_blocking=True), return_features=True
                )
                loss = self.criterion(t, outputs, targets, outputs_old)
                # during training, the usual accuracy is computed on the outputs
                if self.model.is_early_exit():
                    for ic_idx in range(len(outputs)):
                        ic_outputs = outputs[ic_idx]
                        ic_feats = feats[ic_idx]

                        if not self.exemplar_means[0]:
                            hits_taw, hits_tag = self.calculate_metrics(
                                ic_outputs, targets
                            )
                        else:
                            ic_means = self.exemplar_means[ic_idx]
                            ic_feats = pool_nmc_features(
                                ic_feats, ic_idx, len(outputs), self.ic_pooling
                            )
                            hits_taw, hits_tag = self.classify(
                                task=t,
                                features=ic_feats,
                                means=ic_means,
                                targets=targets,
                            )
                        # Log
                        total_loss += loss[ic_idx].item() * len(targets)
                        total_acc_taw[ic_idx] += hits_taw.sum().item()
                        total_acc_tag[ic_idx] += hits_tag.sum().item()
                        total_num += len(targets)
                else:
                    if not self.exemplar_means:
                        hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                    else:
                        hits_taw, hits_tag = self.classify(
                            task=t,
                            features=feats,
                            means=self.exemplar_means,
                            targets=targets,
                        )
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

    def icarl_kd_loss(self, t, outputs, outputs_old):
        g = torch.sigmoid(torch.cat(outputs[:t], dim=1))
        q_i = torch.sigmoid(torch.cat(outputs_old[:t], dim=1))
        loss = self.lamb * sum(
            torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
            for y in range(sum(self.model.task_cls[:t]))
        )
        return loss

    # Algorithm 3: classification and distillation terms -- original formulation has no trade-off parameter (lamb=1)
    def criterion(self, t, outputs, targets, outputs_old=None):
        """Returns the loss value"""

        # Classification loss for new classes
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            loss = []
            for i in range(len(outputs)):
                ic_outputs = outputs[i]
                loss_ce = ic_weights[i] * torch.nn.functional.cross_entropy(
                    torch.cat(ic_outputs, dim=1), targets
                )
                loss_kd = 0
                if t > 0:
                    ic_outputs_old = outputs_old[i]
                    loss_kd = ic_weights[i] * self.icarl_kd_loss(
                        t, ic_outputs, ic_outputs_old
                    )
                loss.append(loss_ce + loss_kd)
            return loss
        else:
            loss_ce = torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets
            )
            # Distillation loss for old classes
            loss_kd = 0
            if t > 0:
                loss_kd = self.icarl_kd_loss(t, outputs, outputs_old)
            return loss_ce + loss_kd

    def ee_net(self, exit_layer: Optional[int] = None) -> torch.nn.Module:
        # early exit network with all CL logic implemented for inference, for profiling
        tmp_model = deepcopy(self.model)
        tmp_model.set_exit_layer(exit_layer)
        icarl_model = iCaRLModelWrapper(
            tmp_model,
            self.exemplar_means,
            logit_conversion=self.nmc_logit_conversion,
            ic_pooling=self.ic_pooling,
        )
        return icarl_model


def pool_nmc_features(features, ic_idx, num_cls, pooling="non"):
    if ic_idx == num_cls - 1 or len(features.shape) != 4:
        return features

    if pooling == "none":
        return features
    elif pooling == "avg":
        return torch.nn.functional.avg_pool2d(features, kernel_size=2)
    elif pooling == "max":
        return torch.nn.functional.max_pool2d(features, kernel_size=2)
    elif pooling == "combined":
        return 0.5 * (
            torch.nn.functional.max_pool2d(features, kernel_size=2)
            + torch.nn.functional.avg_pool2d(features, kernel_size=2)
        )
    else:
        raise NotImplementedError(f"Pooling {pooling} not implemented")


class iCaRLModelWrapper(torch.nn.Module):
    def __init__(
        self, model, exemplar_means, logit_conversion="inverse", ic_pooling="none"
    ):
        super().__init__()
        self.model = model
        self.exemplar_means = exemplar_means
        self.logit_conversion = logit_conversion
        self.ic_pooling = ic_pooling

    def forward(self, x):
        outputs, features = self.model(x, return_features=True)
        if self.model.exit_layer_idx == -1:
            task_sizes = [int(o.shape[-1]) for o in outputs]
            probs = self.classify(features, self.exemplar_means[-1])
            current_task_size = 0
            nmc_outputs = []
            for task_size in task_sizes:
                nmc_outputs.append(
                    probs[:, current_task_size : current_task_size + task_size]
                )
                current_task_size += task_size
            return nmc_outputs
        else:
            icarl_outputs = []
            for ic_idx, (ic_outputs, ic_features) in enumerate(zip(outputs, features)):
                task_sizes = [int(o.shape[-1]) for o in ic_outputs]
                ic_means = self.exemplar_means[ic_idx]
                ic_features = pool_nmc_features(
                    ic_features, ic_idx, len(self.model.ic_layers) + 1, self.ic_pooling
                )
                probs = self.classify(ic_features, ic_means)
                current_task_size = 0
                nmc_outputs = []
                for task_size in task_sizes:
                    nmc_outputs.append(
                        probs[:, current_task_size : current_task_size + task_size]
                    )
                    current_task_size += task_size
                icarl_outputs.append(nmc_outputs)
            return icarl_outputs

    def classify(self, features, means):
        # expand means to all batch images
        means = torch.stack(means)
        means = torch.stack([means] * features.shape[0])
        means = means.transpose(1, 2)
        # expand all features to all classes
        features = features.view(features.shape[0], -1)
        features = features / features.norm(dim=1).view(-1, 1)
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get distances for all images to all exemplar class means -- nearest prototype
        dists = (
            (features - means).pow(2).sum(1)
        )  # TODO removed squeeze - check if it's still ok?
        # TODO do these logits make sense?
        if self.logit_conversion == "inverse":
            logits = 1 / (dists + 10e-6)
        if self.logit_conversion == "reverse":
            logits = -dists
        elif self.logit_conversion == "pdf":
            logits = self.nmc_probs(dists)
        else:
            raise NotImplementedError()
        return logits

    def nmc_probs(self, dists, sigma=1):
        # TODO check this code
        exponent = torch.exp(-0.5 * (dists / sigma**2))
        norm_term = 1 / (2 * torch.pi * sigma**2) ** 0.5
        return norm_term * exponent
