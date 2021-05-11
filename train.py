"""
This example is largely adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

import os
import random
import argparse
import numpy as np
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning.core import LightningModule
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

import intrinsics_utils
from loss_fn import DMPLoss
from dataloader import DepthMotionDataset
from depth_prediction_net import DispNetS
from object_motion_net import MotionVectorNet

rsize_factor = (128,416)

class DepthMotionLightningModel(LightningModule):
    def __init__(self, hparams):
        
        super(DepthMotionLightningModel, self).__init__()
        self.default_loss_weights = {
                            'rgb_consistency': 1.0,
                            'ssim': 3.0,
                            'depth_consistency': 0.05,
                            'depth_smoothing': 0.05,
                            'rotation_cycle_consistency': 1e-3,
                            'translation_cycle_consistency': 5e-2,
                            'depth_variance': 0.0,
                            'motion_smoothing': 1.0,
                            'motion_drift': 0.2,
                        }
        self.hparams = hparams
        self.motion_field_burning_steps = 20000
        self.depth_net = DispNetS()
        intrinsics_mat = None
        if self.hparams.intrinsics:
            intrinsics_mat = np.loadtxt('./intrinsics.txt', delimiter=',')
            intrinsics_mat = intrinsics_mat.reshape(3, 3)
        self.object_motion_net = MotionVectorNet(auto_mask=True, 
                        intrinsics=self.hparams.intrinsics, intrinsics_mat=intrinsics_mat)
        self.loss_func = DMPLoss(self.default_loss_weights)
        self.delete_file = True
        train_batches = len(self.train_dataloader())
        self.base_step = (train_batches) // self.hparams.accumulate_grad_batches
        # torch.autograd.set_detect_anomaly(True)

    def forward(self, x, step, train=False):
        endpoints = {}
        rgb_seq_images = x
        rgb_images = torch.cat((rgb_seq_images[0], rgb_seq_images[1]), dim=0)
        depth_images = self.depth_net(rgb_images)
        depth_seq_images = torch.split(depth_images, depth_images.shape[0] // 2, dim=0)
        endpoints['predicted_depth'] = depth_seq_images
        endpoints['rgb'] = rgb_seq_images
        motion_features = [     
                torch.cat((endpoints['rgb'][0], 
                        endpoints['predicted_depth'][0]), dim=1),
                torch.cat((endpoints['rgb'][0], 
                        endpoints['predicted_depth'][0]), dim=1)]
        motion_features_stack = torch.cat(motion_features, dim=0)
        flipped_motion_features_stack = torch.cat(motion_features[::-1], dim=0)
        pairs = torch.cat([motion_features_stack, 
                        flipped_motion_features_stack], dim=1)
        rot, trans, residual_translation, intrinsics_mat = \
                                        self.object_motion_net(pairs)
        if train and self.motion_field_burning_steps > 0.0:
            step = self.base_step * self.current_epoch
            step = torch.tensor(step).type(torch.FloatTensor)
            burnin_steps = torch.tensor(self.motion_field_burning_steps).type(
                                                    torch.FloatTensor)
            residual_translation *= torch.clamp(2 * step / burnin_steps - 1, 0.0,
                                             1.0)
        endpoints['residual_translation'] = torch.split(residual_translation, 
                            residual_translation.shape[0] // 2, dim=0)
        endpoints['background_translation'] = torch.split(trans, 
                                            trans.shape[0] // 2, dim=0)
        endpoints['rotation'] = torch.split(rot, rot.shape[0] // 2, dim=0)
        intrinsics_mat = 0.5 * sum(
            torch.split(intrinsics_mat, 
                            intrinsics_mat.shape[0] // 2, dim=0))
        endpoints['intrinsics_mat'] = [intrinsics_mat] * 2
        endpoints['intrinsics_mat_inv'] = [
        intrinsics_utils.invert_intrinsics_matrix(intrinsics_mat)] * 2
        return endpoints

    def training_step(self, batch, batch_idx):
        
        endpoints = self.forward(batch, batch_idx, train=True)
        self.logger
        loss_val = self.loss_func(endpoints)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        tqdm_dict = {'train_loss': loss_val}
        outputs = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return outputs

    def validation_step(self, batch, batch_idx):
        
        endpoints = self.forward(batch, batch_idx, train=False)
        loss_val = self.loss_func(endpoints)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
        outputs = OrderedDict({
            'val_loss': loss_val,
        })
        return outputs

    def validation_epoch_end(self, outputs):

        tqdm_dict = {}
        for metric_name in ["val_loss"]:
            metric_total = 0
            for output in outputs:
                metric_value = output[metric_name]
                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)
                metric_total += metric_value
            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = DepthMotionDataset(mode='train', transform=transforms.Compose([
                                        transforms.Resize(size=rsize_factor),
                                        transforms.ToTensor(),
                                    ]),
                                    root_dir='./',
                                    )
        train_loader = torch.utils.data.DataLoader(
                                        dataset=train_dataset,
                                        batch_size=self.hparams.batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        drop_last = False,
                                        sampler=None,
                                        pin_memory=False,
                                    )
        print ("Total train example : {}".format((len(train_loader.dataset))))
        return train_loader

    def val_dataloader(self):
        val_dataset = DepthMotionDataset(mode='valid', transform=transforms.Compose([
                                        transforms.Resize(size=rsize_factor),
                                        transforms.ToTensor(),
                                    ]),
                                    root_dir='./',
                                    )
        val_loader = torch.utils.data.DataLoader(
                                        dataset=val_dataset,
                                        batch_size=self.hparams.batch_size,
                                        shuffle=False,
                                        num_workers=8,
                                        drop_last = False,
                                        sampler=None,
                                        pin_memory=False,
                                    )
        print ("Total valid example : {}".format((len(val_loader.dataset))))
        return val_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--seed', type=int, default=42,
                            help='seed for initializing training. ')
        parser.add_argument('-b', '--batch-size', default=8, type=int,
                            metavar='N',
                            help='mini-batch size (default: 256), this is the total '
                                 'batch size of all GPUs on the current node when '
                                 'using Data Parallel or Distributed Data Parallel')
        parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum')
        parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)',
                            dest='weight_decay')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--intrinsics', dest='intrinsics', action='store_true',
                            help='use specified intrinsics')
        return parser

def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--gpus', type=int, default=0,
                               help='how many gpus')
    parent_parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                               help='supports three options dp, ddp, ddp2')
    parent_parser.add_argument('--use-16bit', dest='use_16bit', action='store_true',
                               help='if true uses 16 bit precision')
    parent_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-cf', '--clear-folder', dest='clear_folder', action='store_true',
                               help='clear the folder')
    parent_parser.add_argument('-agb', '--accumulate-grad-batches', dest='accumulate_grad_batches',type=int,
                                default=4)

    parser = DepthMotionLightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()


def main(hparams):
    
    model = DepthMotionLightningModel(hparams)
   

    if hparams.seed is not None:
        random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        cudnn.deterministic = True
    logger = TensorBoardLogger('./logs') # Log files will be stored in this directory.
    save_path="./checkpoints/" # Checkpoints will be stored in this directory.
    checkpoint_callback = ModelCheckpoint(
                        filepath=os.path.join(save_path, '{val_loss:.3f}'),
                        save_top_k=1,
                        verbose=True,
                        monitor='val_loss',
                        mode='min',
                        period=1
                        )

    trainer = pl.Trainer(
        default_root_dir=save_path,
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.gpus,
        max_epochs=hparams.epochs,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        benchmark=True,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        logger=logger,
        gradient_clip_val=10.0,
    )
    if hparams.evaluate:
        trainer.run_evaluation()
    else:
        trainer.fit(model)

if __name__ == '__main__':    
    main(get_args())