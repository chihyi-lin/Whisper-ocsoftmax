"""A generic training wrapper."""
from copy import deepcopy
import logging
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from src.metrics import calculate_eer, compute_ocsoftmax_eer
from sklearn.metrics import roc_auc_score

"""OCSoftmax function adapted from https://github.com/yzyouzhang/AIR-ASVspoof/blob/master/loss.py"""

class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = nn.Parameter(torch.randn(1, self.feat_dim))
        nn.init.kaiming_uniform_(self.center, 0.25)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0,1)
        output_scores = scores.clone()

        scores[labels == 0] = self.r_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


LOGGER = logging.getLogger(__name__)
file_handler = logging.FileHandler('./trained_models/training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
LOGGER.addHandler(file_handler)

class Trainer:
    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        device: str = "cpu",
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: dict = {"lr": 1e-3},
        use_scheduler: bool = False,
        add_loss: str = None
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler
        self.add_loss = add_loss


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    # get "feat" [8, 16] from model for ocsoftmax instead of batch_out [8, 1]
    feat, batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return feat, batch_out, batch_loss

class GDTrainer(Trainer):

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        test_len: Optional[float] = None,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
    ):
        if test_dataset is not None:
            train = dataset
            test = test_dataset
        else:
            test_len = int(len(dataset) * test_len)
            train_len = len(dataset) - test_len
            lengths = [train_len, test_len]
            train, test = torch.utils.data.random_split(dataset, lengths)

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )

        criterion = torch.nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        if self.add_loss == "ocsoftmax":
            ocsoftmax = OCSoftmax(feat_dim=16).to(self.device)  # feat_dim == feat.size(1)
            ocsoftmax.train()
            ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=self.optimizer_kwargs["lr"])

        LOGGER.info(f"Starting training for {self.epochs} epochs!")

        forward_and_loss_fn = forward_and_loss

        if self.use_scheduler:
            batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optim,
                T_0=batches_per_epoch,
                T_mult=1,
                eta_min=5e-6,
                # verbose=True,
            )
        use_cuda = self.device != "cpu"
        
        best_model = None
        best_loss_model = None
        best_eer = 1e8
        
        for epoch in range(self.epochs):
            LOGGER.info(f"Epoch num: {epoch}")

            running_loss = 0
            # num_correct = 0.0
            num_total = 0.0
            model.train()

            if self.add_loss == None:
                for i, (batch_x, _, batch_y) in enumerate(train_loader):
                    batch_size = batch_x.size(0)
                    num_total += batch_size
                    batch_x = batch_x.to(self.device)

                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                    feat, batch_out, batch_loss = forward_and_loss_fn(
                        model, criterion, batch_x, batch_y, use_cuda=use_cuda
                    )

                    batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                    # num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                    running_loss += batch_loss.item() * batch_size

                    if i % 100 == 0:
                        LOGGER.info(
                            f"[{epoch:04d}][{i:05d}]: {running_loss / num_total}"
                        )
                    optim.zero_grad()
                    batch_loss.backward()
                    optim.step()
                    if self.use_scheduler:
                        scheduler.step()


            elif self.add_loss == "ocsoftmax":
                for i, (batch_x, _, batch_y) in enumerate(train_loader):
                    batch_size = batch_x.size(0)
                    num_total += batch_size
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                    feat, batch_out, batch_loss = forward_and_loss_fn(
                        model, criterion, batch_x, batch_y, use_cuda=use_cuda
                    )
                    ocsoftmaxloss, _ = ocsoftmax(feat, batch_y)
                    batch_loss = ocsoftmaxloss
                    # batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                    # num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

                    running_loss += batch_loss.item() * batch_size

                    if i % 100 == 0:
                        LOGGER.info(
                            f"[{epoch:04d}][{i:05d}]: {running_loss / num_total}"
                        )
                    optim.zero_grad()
                    ocsoftmax_optimzer.zero_grad()
                    batch_loss.backward()
                    optim.step()
                    ocsoftmax_optimzer.step()

                    if self.use_scheduler:
                        scheduler.step()

            running_loss /= num_total
            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: train/loss: {running_loss}"
            )

            test_running_loss = 0.0
            num_total = 0.0
            val_eer = 0
            model.eval()
            y_pred = torch.Tensor([]).to(self.device)
            y = torch.Tensor([]).to(self.device)

            for batch_x, _, batch_y in test_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)

                with torch.no_grad():
                    feat, batch_pred = model(batch_x)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_loss = criterion(batch_pred, batch_y)

                if self.add_loss == None:
                    batch_pred = torch.sigmoid(batch_pred)
                elif self.add_loss == "ocsoftmax":
                    ocsoftmaxloss, batch_pred = ocsoftmax(feat, batch_y)
                    batch_loss = ocsoftmaxloss
                
                test_running_loss += batch_loss.item() * batch_size
                y_pred = torch.concat([y_pred, batch_pred.detach()], dim=0)
                y = torch.concat([y, batch_y.detach()], dim=0)

            if num_total == 0:
                num_total = 1
            test_running_loss /= num_total

            # EER for softmax 
            if self.add_loss == None:
                y_for_eer = 1 - y
                thresh, val_eer, fpr, tpr = calculate_eer(
                    y=y_for_eer.cpu().numpy(),
                    y_score=y_pred.cpu().numpy(),
                )
            # EER for ocsoftmax
            else:
                val_eer, thresh = compute_ocsoftmax_eer(y=y.squeeze(1).cpu().numpy(), y_score=y_pred.cpu().numpy())

            LOGGER.info(
                f"Epoch [{epoch+1}/{self.epochs}]: test/loss: {test_running_loss}, test/eer: {val_eer:.4f}, threshold: {thresh:.4f}"
            )

            if best_model is None or val_eer < best_eer:
                best_eer = val_eer
                best_model = deepcopy(model.state_dict())
                if self.add_loss == "ocsoftmax":
                    best_loss_model = deepcopy(ocsoftmax.state_dict())

            LOGGER.info(
                f"[{epoch:04d}]: train/loss: {running_loss} - test/loss: {test_running_loss} - test/eer: {val_eer:.4f} - threshold: {thresh:.4f}"
            )

        model.load_state_dict(best_model)

        if self.add_loss == "ocsoftmax":
            ocsoftmax.load_state_dict(best_loss_model)

        return model, ocsoftmax
