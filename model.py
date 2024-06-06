import torch
import torchmetrics
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import math
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import config
from train_utils import SegformerForSeg, SegformerForDepth, SegformerForSegDepth
from train_utils import PolyLR
from train_utils import RMSLELoss
from train_utils import compute_depth_metrics

"""
Defines the models
"""

class SegFormer(pl.LightningModule):
    '''
    Model for segmentation only
    '''
    def __init__(self):

        super().__init__()

        pl.seed_everything(42)       # reproducibility

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, semantic_loss_ignore_index = config.IGNORE_INDEX, return_dict=False)
        self.model = SegformerForSeg(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads the weights

        # Initialize the loss
        self.loss = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)

        # Initialize the metrics
        self.iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        #self.calibration_error = torchmetrics.CalibrationError(num_bins=10, task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_index):
        images, labels, _ = batch

        logits = self.model(images)

        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        loss = self.loss(upsampled_logits, labels.squeeze(dim=1).long())
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, labels, _ = batch

        logits = self.model(images)
        
        upsampled_logits = torch.nn.functional.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample logits to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        self.iou(torch.softmax(upsampled_logits, dim=1), labels.squeeze(dim=1))
        self.calibration_error(torch.softmax(upsampled_logits, dim=1), labels.squeeze(dim=1))
        
        self.log('val_iou', self.iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_calibration_error', self.calibration_error, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


class DepthFormer(pl.LightningModule):
    '''
    Model for depth estimation only
    '''
    def __init__(self):

        super().__init__()

        pl.seed_everything(42)       # reproducibility

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=1, semantic_loss_ignore_index = config.IGNORE_INDEX, return_dict=False)
        self.model = SegformerForDepth(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=1, return_dict=False)  # this loads the weights

        # Initialize the loss function
        self.loss = RMSLELoss()

        # Initialize the metrics
        self.metrics = compute_depth_metrics

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_index):
        images, _, depths = batch

        preds = self.model(images) # _ because the model computes the loss internally but we compute it manually below, depths are also merely needed as dummies
        
        preds = F.relu(preds, inplace=True)

        preds = torch.nn.functional.interpolate(preds, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
        
        valid_mask = (depths > 1e-3) & (depths < 10)    # common practice: https://arxiv.org/pdf/2207.04535v2.pdf
        
        loss = self.loss(preds, depths, valid_mask)
        
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    

    def validation_step(self, batch, batch_index):
        images, _, depths = batch

        preds = self.model(images) # _ because the model computes the loss internally but we compute it manually below, depths are also merely needed as dummies
        
        preds = F.relu(preds, inplace=True)

        preds = torch.nn.functional.interpolate(preds, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)

        valid_mask = (depths > 1e-3) & (depths < 10)

        preds = preds[valid_mask]
        depths = depths[valid_mask]

        metrics = self.metrics(preds, depths)
        
        self.log('val_rmse', metrics[0], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_abs_rel', metrics[1], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_log10', metrics[2], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_d1', metrics[3], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d2', metrics[4], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d3', metrics[5], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


class SegDepthFormer(pl.LightningModule):
    '''
    Model for simultaneous segmentation and depth estimation
    '''
    def __init__(self):

        super().__init__()

        pl.seed_everything(42)       # reproducibility

        model_config = SegformerConfig.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, semantic_loss_ignore_index=config.IGNORE_INDEX, return_dict=False)
        self.model = SegformerForSegDepth(model_config) # this does not load the imagenet weights yet.
        self.model = self.model.from_pretrained(f'nvidia/mit-{config.BACKBONE}', num_labels=config.NUM_CLASSES, return_dict=False)  # this loads the weights

        # Initialize the losses
        self.seg_loss = nn.CrossEntropyLoss(ignore_index=config.IGNORE_INDEX)
        self.depth_loss = RMSLELoss()

        # Initialize the metrics
        self.seg_iou = torchmetrics.JaccardIndex(task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        #self.seg_calibration_error = torchmetrics.CalibrationError(num_bins=10, task='multiclass', num_classes=config.NUM_CLASSES, ignore_index=config.IGNORE_INDEX)
        self.depth_metrics = compute_depth_metrics

        # Initialize the optimizer
        self.optimizer = torch.optim.AdamW(params=[
            {'params': self.model.segformer.parameters(), 'lr': config.LEARNING_RATE},
            {'params': self.model.decode_head.parameters(), 'lr': 10 * config.LEARNING_RATE},
            {'params': self.model.depth_head.parameters(), 'lr': 10 * config.LEARNING_RATE}
        ], lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)


    def configure_optimizers(self):
        iterations_per_epoch = math.ceil(config.NUMBER_TRAIN_IMAGES / (config.BATCH_SIZE * len(config.DEVICES)))
        total_iterations = iterations_per_epoch * self.trainer.max_epochs
        scheduler = PolyLR(self.optimizer, max_iterations=total_iterations, power=0.9)
        return [self.optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_index):
        images, labels, depths = batch

        seg_logits, depth_preds = self.model(images)
        
        seg_logits = torch.nn.functional.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
        depth_preds = F.relu(depth_preds, inplace=True) 
        depth_preds = torch.nn.functional.interpolate(depth_preds, size=images.shape[-2:], mode="bilinear", align_corners=False)

        valid_mask = (depths > 1e-3) & (depths < 10)    # common practice: https://arxiv.org/pdf/2207.04535v2.pdf
        
        seg_loss = self.seg_loss(seg_logits, labels.squeeze(dim=1))
        depth_loss = self.depth_loss(depth_preds, depths, valid_mask)
        total_loss = config.LOSS_SEG_WEIGHT * seg_loss + config.LOSS_DEPTH_WEIGHT * depth_loss

        self.log('seg_loss', seg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('depth_loss', depth_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('total_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return total_loss
    

    def validation_step(self, batch, batch_index):
        images, labels, depths = batch

        seg_logits, depth_preds = self.model(images)
        seg_logits = torch.nn.functional.interpolate(seg_logits, size=images.shape[-2:], mode="bilinear", align_corners=False)    # upsample preds to input image size (SegFormer outputs h/4 and w/4 by default, see paper)
        depth_preds = torch.nn.functional.interpolate(depth_preds, size=images.shape[-2:], mode="bilinear", align_corners=False)   
        depth_preds = F.relu(depth_preds, inplace=True) 
        
        valid_mask = (depths > 1e-3) & (depths < 10)    # common practice: https://arxiv.org/pdf/2207.04535v2.pdf
        depth_preds = depth_preds[valid_mask]
        depths = depths[valid_mask]

        self.seg_iou(torch.softmax(seg_logits, dim=1), labels.squeeze(dim=1))
        self.seg_calibration_error(torch.softmax(seg_logits, dim=1), labels.squeeze(dim=1))
        depth_metrics = compute_depth_metrics(depth_preds, depths)

        self.log('val_iou', self.seg_iou, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_calibration_error', self.seg_calibration_error, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_rmse', depth_metrics[0], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_abs_rel', depth_metrics[1], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_log10', depth_metrics[2], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('val_d1', depth_metrics[3], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d2', depth_metrics[4], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_d3', depth_metrics[5], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)