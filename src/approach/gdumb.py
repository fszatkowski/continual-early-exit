from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from datasets.exemplars_dataset import ExemplarsDataset

from .incremental_learning import Inc_Learning_Appr


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


class Appr(Inc_Learning_Appr):
    """Class implementing the GDumb approach
    described in https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470511.pdf
    Original code available at https://github.com/drimpossible/GDumb
    """

    def __init__(
        self,
        model,
        device,
        nepochs=100,
        lr=0.05,
        lr_min=0.0005,
        lr_factor=3,
        lr_patience=5,
        clipgrad=10.0,
        momentum=0.9,
        wd=1e-6,
        multi_softmax=False,
        fix_bn=False,
        eval_on_train=False,
        select_best_model_by_val_loss=True,
        logger=None,
        exemplars_dataset=None,
        scheduler_milestones=None,
        regularization="cutmix",
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
        self.regularization = regularization
        self.init_model = None

        have_exemplars = (
            self.exemplars_dataset.max_num_exemplars
            + self.exemplars_dataset.max_num_exemplars_per_class
        )
        assert have_exemplars > 0, "Error: GDumb needs exemplars."

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--regularization",
            default="cutmix",
            required=False,
            help="Use regularization (default=%(default)s)",
        )
        return parser.parse_known_args(args)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        # 1. GDumb resets the network before learning a new task, relying only on the exemplars stored so far
        if t == 0:
            # Keep the randomly initialized model from time step 0
            self.init_model = self.model.state_dict()
        else:
            # Reinitialize the model (backbone) for very task from time step 1
            self.model.load_state_dict(self.init_model, strict=False)

        # EXEMPLAR MANAGEMENT -- select training subset from current task and exemplar memory
        aux_loader = torch.utils.data.DataLoader(
            trn_loader.dataset + self.exemplars_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )
        self.exemplars_dataset.collect_exemplars(
            self.model, aux_loader, val_loader.dataset.transform
        )

        # Set new set of exemplars as the only data to train on
        trn_loader = torch.utils.data.DataLoader(
            self.exemplars_dataset,
            batch_size=trn_loader.batch_size,
            shuffle=True,
            num_workers=trn_loader.num_workers,
            pin_memory=trn_loader.pin_memory,
        )

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Get exemplars
            if len(self.exemplars_dataset) > 0:
                # 2. Balanced batches
                exemplar_indices = torch.randperm(len(self.exemplars_dataset))[
                    : trn_loader.batch_size
                ]
                images_exemplars, targets_exemplars = default_collate(
                    [self.exemplars_dataset[i] for i in exemplar_indices]
                )
                images = torch.cat((images, images_exemplars), dim=0)
                targets = torch.cat((targets, targets_exemplars), dim=0)

            # 3. Apply cutmix as regularization
            do_cutmix = (
                self.regularization == "cutmix" and np.random.rand(1) < 0.5
            )  # cutmix_prob (Sec.4)

            if t == 1:
                print("???")

            if do_cutmix:
                images, targets_a, targets_b, lamb = cutmix_data(
                    x=images, y=targets, alpha=1.0
                )  # cutmix_alpha (Sec.4)
                # Forward current model
                outputs = self.model(images.to(self.device, non_blocking=True))

                if self.model.is_early_exit():
                    losses_a = self.criterion(
                        t, outputs, targets_a.to(self.device, non_blocking=True)
                    )
                    losses_b = self.criterion(
                        t, outputs, targets_b.to(self.device, non_blocking=True)
                    )
                    losses = [
                        lamb * l_a + (1.0 - lamb) * l_b
                        for l_a, l_b in zip(losses_a, losses_b)
                    ]
                    loss = sum(losses)
                else:
                    loss = lamb * self.criterion(
                        t, outputs, targets_a.to(self.device, non_blocking=True)
                    )
                    loss += (1.0 - lamb) * self.criterion(
                        t, outputs, targets_b.to(self.device, non_blocking=True)
                    )
            else:
                outputs = self.model(images.to(self.device, non_blocking=True))
                loss = self.criterion(
                    t, outputs, targets.to(self.device, non_blocking=True)
                )
                if self.model.is_early_exit():
                    loss = sum(loss)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        if self.model.is_early_exit():
            ic_weights = self.model.get_ic_weights(
                current_epoch=self.current_epoch, max_epochs=self.nepochs
            )
            ic_losses = []
            for ic_weight, ic_output in zip(ic_weights, outputs):
                ic_loss = torch.nn.functional.cross_entropy(
                    torch.cat(ic_output, dim=1), targets
                )
                ic_losses.append(ic_weight * ic_loss)
            return ic_losses
        else:
            return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
