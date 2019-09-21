import sys

import cv2
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import transforms

from h5.dataset import TrainDataset, ValidateDataset

# from .data.dataset_loader import (TrainDataset, TrainDatasetParam,
#                                   ValidateDataset)
from .data.image_reader import ImageReader
from .kpi.psnr import PSNRNdarray
from .logger import get_logger
from .loss.total_variation import TotalVariationMSELoss
from .model.mch import MCH
from .trainer.sample import Trainer

torch.manual_seed(0)

logger = get_logger(__file__)

scale = 2
train_path = '/Set5/y'
test_path = '/Set5/y'
gpu_device = 'cuda'
num_epoch = 1000


# train_dataset = ImageReader(train_path)
# test_dataset = ImageReader(test_path)

# train_transform = transforms.Compose([transforms.ToTensor()])

# train_dataset_param = TrainDatasetParam(15, 5, 2, input_size=30)
# train_loader = TrainDataset(train_dataset, train_dataset_param, train_transform)
# train_loader = torch.utils.data.DataLoader(dataset=train_loader, batch_size=32, shuffle=True)

# test_loader = ValidateDataset(test_dataset, 2, 5, train_transform)
# test_loader = torch.utils.data.DataLoader(dataset=test_loader, batch_size=1, shuffle=False)

train_dataset = TrainDataset(f'{os.path.dirname(__file__)}/h5/train.h5')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=1,
                              pin_memory=True,
                              drop_last=True)
eval_dataset = ValidateDataset(f'{os.path.dirname(__file__)}/h5/test.h5')
test_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)


net = MCH(scale, 1)

lr_params = [{'params': net.conv1.weight, 'lr': 0.0001},
             #  {'params': net.conv1.bias, 'lr': 0.00001},
             {'params': net.conv2.weight, 'lr': 0.0001},
             #  {'params': net.conv2.bias, 'lr': 0.00001},
             {'params': net.conv3.weight, 'lr': 0.00001}, ]
#  {'params': net.conv3.bias, 'lr': 0.00001}]

optimizer = optim.Adam(params=lr_params)
optimizer = optim.Adam(net.parameters(), lr=0.0001)

losses = [torch.nn.MSELoss(), TotalVariationMSELoss()]
weights = [1.0, 0.0]

trainer = Trainer(train_loader, test_loader, net, optimizer, losses,
                  num_epoch, gpu_device=gpu_device, loss_weights=weights)


psnr_calculator = PSNRNdarray()


def minmax_numpy(src):
    max = np.where(src < 1, src, 1)
    return np.where(max > 0, max, 0)


logger.info('train')
for epoch in range(num_epoch):

    for batch_idx, (cnn_inputs, cnn_labels) in enumerate(train_loader):
        cnn_inputs = Variable(cnn_inputs).to(gpu_device)
        cnn_labels = Variable(cnn_labels).to(gpu_device)

        optimizer.zero_grad()

        cnn_outputs = net(cnn_inputs)

        weighted_total_loss, losses = trainer._calc_losses(cnn_outputs, cnn_labels)

        weighted_total_loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write(
            f'Epoch [{epoch+1}/{num_epoch}], Batch: [{batch_idx+1}/{len(train_loader)}]l2_loss: {losses[0]:.6f}, tv_loss: {losses[1]:.6f}')
        sys.stdout.flush()

    psnrs = []
    psnrs2 = []
    for batch_idx, (cnn_inputs, cnn_labels) in enumerate(test_loader):
        net.eval()
        with torch.no_grad():
            cnn_inputs = Variable(cnn_inputs).to(gpu_device)
            cnn_labels = Variable(cnn_labels).to(gpu_device)
            cnn_outputs = net(cnn_inputs)
            weighted_total_loss, losses = (cnn_outputs, cnn_labels)

            img = cnn_outputs.cpu().numpy() * 255
            if img.shape[0] == 1:
                img = np.squeeze(img)
            img_uint8 = img.astype(np.uint8)
            cv2.imwrite(f'0.bmp', img_uint8)
            ref = cnn_labels.cpu().numpy() * 255
            if ref.shape[0] == 1:
                ref = np.squeeze(ref)
            ref_uint8 = ref.astype(np.uint8)
            cv2.imwrite(f'1.bmp', ref_uint8)

            psnr = psnr_calculator.calc_psnr(img_uint8, ref_uint8)
            psnr2 = psnr_calculator.calc_psnr(minmax_numpy(np.squeeze(
                cnn_outputs.cpu().numpy())), minmax_numpy(np.squeeze(cnn_labels.cpu().numpy())))
            psnrs.append(psnr)
            psnrs2.append(psnr2)
    print("\n", sum(psnrs) / len(psnrs))
    print(sum(psnrs2) / len(psnrs2))
