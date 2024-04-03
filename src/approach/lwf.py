from argparse import ArgumentParser
from copy import deepcopy

import torch

from datasets.exemplars_dataset import ExemplarsDataset
from metrics import cka

from .incremental_learning import Inc_Learning_Appr


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning Without Forgetting (LwF) approach
    described in https://arxiv.org/abs/1606.09282
    """

    # Weight decay of 0.0005 is used in the original article (page 4).
    # Page 4: "The warm-up step greatly enhances fine-tuning’s old-task performance, but is not so crucial to either our
    #  method or the compared Less Forgetting Learning (see Table 2(b))."
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
        lamb=1,
        T=2,
        mc=False,
        taskwise_kd=False,
        cka=False,
        debug_loss=False,
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
        self.T = T
        self.mc = mc
        self.taskwise_kd = taskwise_kd

        self.cka = cka
        self.debug_loss = debug_loss

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Page 5: "lambda is a loss balance weight, set to 1 for most our experiments. Making lambda larger will favor
        # the old task performance over the new task’s, so we can obtain a old-task-new-task performance line by
        # changing lambda."
        parser.add_argument(
            "--lamb",
            default=1,
            type=float,
            required=False,
            help="Forgetting-intransigence trade-off (default=%(default)s)",
        )
        # Page 5: "We use T=2 according to a grid search on a held out set, which aligns with the authors’
        #  recommendations." -- Using a higher value for T produces a softer probability distribution over classes.
        parser.add_argument(
            "--T",
            default=2,
            type=int,
            required=False,
            help="Temperature scaling (default=%(default)s)",
        )
        parser.add_argument(
            "--mc",
            default=False,
            action="store_true",
            required=False,
            help="If set, will use LwF.MC variant from iCaRL. (default=%(default)s)",
        )
        parser.add_argument(
            "--taskwise-kd",
            default=False,
            action="store_true",
            required=False,
            help="If set, will use task-wise KD loss as defined in SSIL. (default=%(default)s)",
        )

        parser.add_argument(
            "--debug-loss",
            default=False,
            action="store_true",
            required=False,
            help="If set, will log intermediate loss values. (default=%(default)s)",
        )

        return parser.parse_known_args(args)

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

        self.training = True
        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)
        self.training = False

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(
            self.model, trn_loader, val_loader.dataset.transform
        )

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()

        for images, targets in trn_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            # Forward old model
            targets_old = None
            if t > 0:
                targets_old = self.model_old(images)

            # Forward current model
            outputs = self.model(images)
            if self.debug_loss:
                loss, loss_kd, loss_ce = self.criterion(
                    t, outputs, targets, targets_old, return_partial_losses=True
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_kd",
                    group=f"debug_t{t}",
                    value=float(loss_kd),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_ce",
                    group=f"debug_t{t}",
                    value=float(loss_ce),
                )
                self.logger.log_scalar(
                    task=None,
                    iter=None,
                    name="loss_total",
                    group=f"debug_t{t}",
                    value=float(loss),
                )
            else:
                loss = self.criterion(
                    t, outputs, targets, targets_old, return_partial_losses=False
                )

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
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            if self.model_old is not None:
                self.model_old.eval()

            for images, targets in val_loader:
                images, targets = images.to(self.device, non_blocking=True), targets.to(
                    self.device, non_blocking=True
                )
                # Forward old model
                targets_old = None
                if t > 0:
                    targets_old = self.model_old(images)
                # Forward current model
                outputs = self.model(images)
                loss = self.criterion(t, outputs, targets, targets_old)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.data.cpu().numpy().item() * len(targets)
                total_acc_taw += hits_taw.sum().data.cpu().numpy().item()
                total_acc_tag += hits_tag.sum().data.cpu().numpy().item()
                total_num += len(targets)

        if self.cka and t > 0 and self.training:
            _cka = cka(self.model, self.model_old, val_loader, self.device)
            self.logger.log_scalar(
                task=None, iter=None, name=f"t_{t}", group="cka", value=_cka
            )

        return (
            total_loss / total_num,
            total_acc_taw / total_num,
            total_acc_tag / total_num,
        )

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        """Calculates cross-entropy with temperature scaling"""
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(
        self, t, outputs, targets, outputs_old=None, return_partial_losses=False
    ):
        """Returns the loss value"""
        if t > 0 and outputs_old is not None:
            # Knowledge distillation loss for all previous tasks
            kd_outputs, kd_outputs_old = torch.cat(outputs[:t], dim=1), torch.cat(
                outputs_old[:t], dim=1
            )

            if self.mc:
                g = torch.sigmoid(kd_outputs)
                q_i = torch.sigmoid(kd_outputs_old)
                loss_kd = sum(
                    torch.nn.functional.binary_cross_entropy(g[:, y], q_i[:, y])
                    for y in range(kd_outputs.shape[-1])
                )
            elif self.taskwise_kd:
                loss_kd = torch.zeros(t).to(self.device, non_blocking=True)
                for _t in range(t):
                    soft_target = torch.nn.functional.softmax(
                        outputs_old[_t] / self.T, dim=1
                    )
                    output_log = torch.nn.functional.log_softmax(
                        outputs[_t] / self.T, dim=1
                    )
                    loss_kd[_t] = torch.nn.functional.kl_div(
                        output_log, soft_target, reduction="batchmean"
                    ) * (self.T**2)
                loss_kd = loss_kd.sum()
            else:
                loss_kd = self.cross_entropy(
                    kd_outputs, kd_outputs_old, exp=1.0 / self.T
                )
        else:
            loss_kd = 0

        # Current cross-entropy loss -- with exemplars use all heads
        if len(self.exemplars_dataset) > 0:
            loss_ce = torch.nn.functional.cross_entropy(
                torch.cat(outputs, dim=1), targets
            )
        else:
            loss_ce = torch.nn.functional.cross_entropy(
                outputs[t], targets - self.model.task_offset[t]
            )

        if self.lamb is None:
            self.lamb = 0.0
        if return_partial_losses:
            return self.lamb * loss_kd + loss_ce, loss_kd, loss_ce
        else:
            return self.lamb * loss_kd + loss_ce
