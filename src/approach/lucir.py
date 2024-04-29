import copy
import math
import warnings
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module, Parameter

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(
        self,
        model,
        device,
        nepochs=160,
        lr=0.1,
        lr_min=1e-4,
        lr_factor=10,
        lr_patience=8,
        clipgrad=10000,
        momentum=0.9,
        wd=5e-4,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger=None,
        exemplars_dataset=None,
        scheduler_milestones=None,
        lamb=5.0,
        lamb_mr=1.0,
        dist=0.5,
        K=2,
        remove_less_forget=False,
        remove_margin_ranking=False,
        remove_adapt_lamda=False,
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
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda

        self.lamda = self.lamb
        self.ref_model = None

        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        if not have_exemplars:
            warnings.warn(
                "Warning: LUCIR is expected to use exemplars. Check documentation."
            )

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument(
            "--lamb",
            default=5.0,
            type=float,
            required=False,
            help="Trade-off for distillation loss (default=%(default)s)",
        )
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument(
            "--lamb-mr",
            default=1.0,
            type=float,
            required=False,
            help="Trade-off for the MR loss (default=%(default)s)",
        )
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument(
            "--dist",
            default=0.5,
            type=float,
            required=False,
            help="Margin threshold for the MR loss (default=%(default)s)",
        )
        # Sec 4.1: "K is set to 2"
        parser.add_argument(
            "--K",
            default=2,
            type=int,
            required=False,
            help='Number of "new class embeddings chosen as hard negatives '
            "for MR loss (default=%(default)s)",
        )
        # Flags for ablating the approach
        parser.add_argument(
            "--remove-less-forget",
            action="store_true",
            required=False,
            help="Deactivate Less-Forget loss constraint(default=%(default)s)",
        )
        parser.add_argument(
            "--remove-margin-ranking",
            action="store_true",
            required=False,
            help="Deactivate Inter-Class separation loss constraint (default=%(default)s)",
        )
        parser.add_argument(
            "--remove-adapt-lamda",
            action="store_true",
            required=False,
            help="Deactivate adapting lambda according to the number of classes (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if self.less_forget:
            # Don't update heads when Less-Forgetting constraint is activated (from original code):
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

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == "ResNet":
                last_block_name = None
                last_layer_idx = 1
                while hasattr(self.model.model, f"layer{last_layer_idx + 1}"):
                    last_layer_idx += 1
                print(f"Replacing last layer at layer{last_layer_idx}...")
                last_layer = getattr(self.model.model, f"layer{last_layer_idx}")
                old_block = last_layer[-1]
                if old_block.__class__.__name__ != "BasicBlock":
                    warnings.warn("Final resnet layer is not a BasicBlock instance")
                new_block = BasicBlockNoRelu(
                    old_block.conv1,
                    old_block.bn1,
                    old_block.relu,
                    old_block.conv2,
                    old_block.bn2,
                    old_block.downsample,
                )
                last_layer = getattr(self.model.model, f"layer{last_layer_idx}")
                last_layer[-1] = new_block
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        if self.model.is_early_exit():
            n_classifiers = len(self.model.heads)
            for n_cls in range(n_classifiers):
                # Apply pooling in case of internal classifiers
                cls_head = self.model.heads[n_cls][-1]
                if isinstance(cls_head, nn.Linear):
                    pooling = False
                    in_features = cls_head.in_features
                    out_features = cls_head.out_features
                else:
                    pooling = True
                    in_features = cls_head.classifier.in_features
                    out_features = cls_head.classifier.out_features
                cosine_head = CosineLinear(
                    in_features=in_features, out_features=out_features, pooling=pooling
                )
                self.model.heads[n_cls][-1] = cosine_head
        else:
            self.model.heads[-1] = CosineLinear(
                self.model.heads[-1].in_features, self.model.heads[-1].out_features
            )

        self.model.to(self.device, non_blocking=True)
        if t > 0:
            if self.model.is_early_exit():
                n_classifiers = len(self.model.heads)
                for n_cls in range(n_classifiers - 1):
                    # Share sigma (Eta in paper) between all the heads
                    self.model.heads[n_cls][-1].sigma = self.model.heads[n_cls][
                        -2
                    ].sigma
                    # Fix previous heads when Less-Forgetting constraint is activated (from original code)
                    if self.less_forget:
                        for h in self.model.heads[n_cls][:-1]:
                            for param in h.parameters():
                                param.requires_grad = False
                        self.model.heads[n_cls][-1].sigma.requires_grad = True
                # Eq. 7: Adaptive lambda
                if self.adapt_lamda:
                    self.lamda = self.lamb * math.sqrt(
                        sum(
                            [
                                h.out_features
                                for h in self.model.heads[n_classifiers - 1][:-1]
                            ]
                        )
                        / self.model.heads[n_classifiers - 1][-1].out_features
                    )
            else:
                # Share sigma (Eta in paper) between all the heads
                self.model.heads[-1].sigma = self.model.heads[-2].sigma
                # Fix previous heads when Less-Forgetting constraint is activated (from original code)
                if self.less_forget:
                    for h in self.model.heads[:-1]:
                        for param in h.parameters():
                            param.requires_grad = False
                    self.model.heads[-1].sigma.requires_grad = True
                # Eq. 7: Adaptive lambda
                if self.adapt_lamda:
                    self.lamda = self.lamb * math.sqrt(
                        sum([h.out_features for h in self.model.heads[:-1]])
                        / self.model.heads[-1].out_features
                    )
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

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
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        if self.model.is_early_exit():
            for cls_heads in self.ref_model.heads:
                for task_head in cls_heads:
                    task_head.train()
        else:
            for h in self.ref_model.heads:
                h.train()
        self.ref_model.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            images, targets = images.to(self.device, non_blocking=True), targets.to(
                self.device, non_blocking=True
            )
            # Forward current model
            outputs, features = self.model(images, return_features=True)
            # Forward previous model
            ref_outputs = None
            ref_features = None
            if t > 0:
                ref_outputs, ref_features = self.ref_model(images, return_features=True)
            loss = self.criterion(
                t, outputs, targets, ref_outputs, features, ref_features
            )
            if self.model.is_early_exit():
                loss = sum(loss)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def criterion(
        self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None
    ):
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            n_classifiers = len(self.model.heads)

        """Returns the loss value"""
        if ref_outputs is None or ref_features is None:
            if self.model.is_early_exit():
                loss = []
                for ic_idx in range(n_classifiers):
                    ic_outputs = outputs[ic_idx]
                    if type(ic_outputs[0]) == dict:
                        ic_outputs = torch.cat([o["wsigma"] for o in ic_outputs], dim=1)
                    else:
                        ic_outputs = torch.cat(ic_outputs, dim=1)
                    ic_loss = ic_weights[ic_idx] * nn.CrossEntropyLoss(None)(
                        ic_outputs, targets
                    )
                    loss.append(ic_loss)
            else:
                if type(outputs[0]) == dict:
                    outputs = torch.cat([o["wsigma"] for o in outputs], dim=1)
                else:
                    outputs = torch.cat(outputs, dim=1)
                # Eq. 1: regular cross entropy
                loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                if self.model.is_early_exit():
                    loss_dist = []
                    for ic_idx in range(n_classifiers):
                        ic_features = features[ic_idx]
                        ic_ref_features = ref_features[ic_idx]
                        ic_features = ic_features.view(ic_features.shape[0], -1)
                        ic_ref_features = ic_ref_features.view(
                            ic_ref_features.shape[0], -1
                        )
                        ic_loss_dist = self.lamda * nn.CosineEmbeddingLoss()(
                            ic_features,
                            ic_ref_features.detach(),
                            torch.ones(targets.shape[0]).to(
                                self.device, non_blocking=True
                            ),
                        )
                        loss_dist.append(ic_weights[ic_idx] * ic_loss_dist)
                else:
                    loss_dist = (
                        nn.CosineEmbeddingLoss()(
                            features,
                            ref_features.detach(),
                            torch.ones(targets.shape[0]).to(
                                self.device, non_blocking=True
                            ),
                        )
                        * self.lamda
                    )
            else:
                if self.model.is_early_exit():
                    loss_dist = []
                    for ic_idx in range(n_classifiers):
                        ic_ref_outputs = ref_outputs[ic_idx]
                        ic_outputs = outputs[ic_idx]
                        # Scores before scale, [-1, 1]
                        ic_ref_outputs = torch.cat(
                            [ro["wosigma"] for ro in ic_ref_outputs], dim=1
                        ).detach()
                        ic_old_scores = torch.cat(
                            [o["wosigma"] for o in ic_outputs[:-1]], dim=1
                        )
                        num_old_classes = ic_ref_outputs.shape[1]

                        # Eq. 5: Modified distillation loss for cosine normalization
                        ic_loss_dist = (
                            nn.MSELoss()(ic_old_scores, ic_ref_outputs)
                            * self.lamda
                            * num_old_classes
                        )
                        loss_dist.append(ic_weights[ic_idx] * ic_loss_dist)
                else:
                    # Scores before scale, [-1, 1]
                    ref_outputs = torch.cat(
                        [ro["wosigma"] for ro in ref_outputs], dim=1
                    ).detach()
                    old_scores = torch.cat([o["wosigma"] for o in outputs[:-1]], dim=1)
                    num_old_classes = ref_outputs.shape[1]

                    # Eq. 5: Modified distillation loss for cosine normalization
                    loss_dist = (
                        nn.MSELoss()(old_scores, ref_outputs)
                        * self.lamda
                        * num_old_classes
                    )

            if self.model.is_early_exit():
                loss_mr = [
                    torch.zeros(1).to(self.device, non_blocking=True)
                    for _ in range(n_classifiers)
                ]
            else:
                loss_mr = torch.zeros(1).to(self.device, non_blocking=True)

            if self.margin_ranking:
                if self.model.is_early_exit():
                    for ic_idx in range(n_classifiers):
                        ic_outputs = outputs[ic_idx]
                        # Scores before scale, [-1, 1]
                        ic_outputs_wos = torch.cat(
                            [o["wosigma"] for o in ic_outputs], dim=1
                        )
                        ic_num_old_classes = (
                            ic_outputs_wos.shape[1] - ic_outputs[-1]["wosigma"].shape[1]
                        )

                        # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                        # The index of hard samples, i.e., samples from old classes
                        ic_hard_index = targets < ic_num_old_classes
                        ic_hard_num = ic_hard_index.sum()

                        if ic_hard_num > 0:
                            # Get "ground truth" scores
                            ic_gt_scores = ic_outputs_wos.gather(
                                1, targets.unsqueeze(1)
                            )[ic_hard_index]
                            ic_gt_scores = ic_gt_scores.repeat(1, self.K)

                            # Get top-K scores on novel classes
                            ic_max_novel_scores = ic_outputs_wos[
                                ic_hard_index, ic_num_old_classes:
                            ].topk(self.K, dim=1)[0]

                            assert ic_gt_scores.size() == ic_max_novel_scores.size()
                            assert ic_gt_scores.size(0) == ic_hard_num
                            # Eq. 8: margin ranking loss
                            ic_loss_mr = nn.MarginRankingLoss(margin=self.dist)(
                                ic_gt_scores.view(-1),
                                ic_max_novel_scores.view(-1),
                                torch.ones(ic_hard_num * self.K).to(
                                    self.device, non_blocking=True
                                ),
                            )
                            loss_mr[ic_idx] = (
                                ic_weights[ic_idx] * self.lamb_mr * ic_loss_mr
                            )
                else:
                    # Scores before scale, [-1, 1]
                    outputs_wos = torch.cat([o["wosigma"] for o in outputs], dim=1)
                    num_old_classes = (
                        outputs_wos.shape[1] - outputs[-1]["wosigma"].shape[1]
                    )

                    # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                    # The index of hard samples, i.e., samples from old classes
                    hard_index = targets < num_old_classes
                    hard_num = hard_index.sum()

                    if hard_num > 0:
                        # Get "ground truth" scores
                        gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[
                            hard_index
                        ]
                        gt_scores = gt_scores.repeat(1, self.K)

                        # Get top-K scores on novel classes
                        max_novel_scores = outputs_wos[
                            hard_index, num_old_classes:
                        ].topk(self.K, dim=1)[0]

                        assert gt_scores.size() == max_novel_scores.size()
                        assert gt_scores.size(0) == hard_num
                        # Eq. 8: margin ranking loss
                        loss_mr = nn.MarginRankingLoss(margin=self.dist)(
                            gt_scores.view(-1),
                            max_novel_scores.view(-1),
                            torch.ones(hard_num * self.K).to(
                                self.device, non_blocking=True
                            ),
                        )
                        loss_mr *= self.lamb_mr

            if self.model.is_early_exit():
                # Eq. 1: regular cross entropy
                loss_ce = []
                for ic_idx in range(n_classifiers):
                    ic_outputs = outputs[ic_idx]
                    ic_loss_ce = nn.CrossEntropyLoss()(
                        torch.cat([o["wsigma"] for o in ic_outputs], dim=1), targets
                    )
                    loss_ce.append(ic_weights[ic_idx] * ic_loss_ce)
                # Eq. 9: integrated objective per classifier
                loss = [
                    loss_dist[ic_idx] + loss_ce[ic_idx] + loss_mr[ic_idx]
                    for ic_idx in range(n_classifiers)
                ]
            else:
                # Eq. 1: regular cross entropy
                loss_ce = nn.CrossEntropyLoss()(
                    torch.cat([o["wsigma"] for o in outputs], dim=1), targets
                )
                # Eq. 9: integrated objective
                loss = loss_dist + loss_ce + loss_mr

        return loss


# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True, pooling=False):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooling = pooling
        if pooling:
            # TODO this is tailored for SDN type of IC
            self.maxpool = nn.MaxPool2d(kernel_size=2)
            self.avgpool = nn.AvgPool2d(kernel_size=2)
            self.alpha = nn.Parameter(torch.rand(1).squeeze())
        else:
            self.register_parameter("alpha", None)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter("sigma", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma
        if self.alpha is not None:
            self.alpha.data.fill_(torch.rand(1).squeeze())

    def forward(self, input):
        if self.pooling:
            input = self.alpha * self.maxpool(input) + (1 - self.alpha) * self.avgpool(
                input
            )
            input = input.view(input.size(0), -1)
        out = F.linear(
            F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1)
        )
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {"wsigma": out_s, "wosigma": out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out
