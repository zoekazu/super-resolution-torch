
import sys
from typing import List, Tuple

import torch
from torch.autograd import Variable

from ..logger import get_logger
from .meta import TrainerMeta


class Trainer(TrainerMeta):
    def __init__(self, train_loader, validate_loaders, net,
                 optimizer, losses, total_epoch, gpu_device=None, loss_weights=None):
        self._train_loader = train_loader
        self._validate_loaders = validate_loaders
        self._optimizer = optimizer
        self._losses = losses
        self._total_epoch = total_epoch
        self._gpu_device = gpu_device
        if gpu_device:
            self._net = net.to(gpu_device)
        else:
            self._net = net
        if loss_weights:
            self._loss_weights = loss_weights
        else:
            self._loss_weights = [1] * len(losses)

        self._logger = get_logger(__file__)

    def _calc_losses(self, cnn_outputs, cnn_labels):
        losses = []
        for loss_idx, (loss, weight) in enumerate(zip(self._losses, self._loss_weights)):
            loss_ = loss(cnn_outputs, cnn_labels)
            if loss_idx == 0:
                weighted_total_loss = loss_ * weight
            else:
                weighted_total_loss += loss_ * weight
            losses.append(loss_.item())
        return weighted_total_loss, losses

    def _batch_calc(self, loader, epoch: int, is_train=False):
        for batch_idx, (cnn_inputs, cnn_labels) in enumerate(loader):
            cnn_inputs = Variable(cnn_inputs).to(self._gpu_device)
            cnn_labels = Variable(cnn_labels).to(self._gpu_device)

            if is_train:
                self._optimizer.zero_grad()

            cnn_outputs = self._net(cnn_inputs)

            weighted_total_loss, losses = self._calc_losses(cnn_outputs, cnn_labels)
            if is_train:
                weighted_total_loss.backward()
                self._optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write(
                    f'Epoch [{epoch+1}/{self._total_epoch}], Batch: [{batch_idx+1}/{len(loader)}]l2_loss: {losses[0]:.6f}, entropy_loss: {losses[1]:.6f}')
                sys.stdout.flush()

        return weighted_total_loss, losses

    def train(self, epoch: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._net.train()
        return self._batch_calc(self._train_loader, epoch, is_train=True)

    def validate(self, epoch: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        self._net.eval()
        with torch.no_grad():
            return self._batch_calc(self._train_loader, epoch, is_train=False)

    def get_outputs(self, cnn_inputs, cnn_outputs, is_train=False):
        if is_train:
            self
        return

    def set_train_mode(self):
        self._net.train()

    def set_validate_mode(self):
        self._net.eval()