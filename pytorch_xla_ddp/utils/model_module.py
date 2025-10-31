import timm
import torch
import torch.nn as nn
from torchmetrics import Metric
from torch.optim.lr_scheduler import LambdaLR
from utils.data_module import divide_no_nan
import numpy as np


class Conv2dBlock(nn.Module):
    """
    Builds a convolutional block with two Conv2D layers, optional BatchNormalization, 
    and ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        use_bn: If True, applies Batch Normalization after each convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, use_bn: bool):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_bn:
            layers.insert(1, nn.BatchNorm2d(out_channels))
            layers.insert(-1, nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    A decoder block that performs upsampling using transposed convolution,
    concatenates with encoder features (skip connection), and applies a Conv2dBlock.

    Args:
        dec_channels: Number of channels for the block's output and the upsampled features.
        enc_channels: Number of channels from the encoder feature map.
        stride: Upsampling factor for transposed convolution. 
        padding: Padding for transposed convolution. 
        use_bn: If True, applies Batch Normalization in the convolutional block. 
    """
    
    def __init__(
        self, 
        dec_channels: int, 
        enc_channels: int, 
        stride: int = 2, 
        padding: int = 1, 
        use_bn: bool = True
    ):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(
            dec_channels * 2, dec_channels, kernel_size=3, stride=stride, padding=padding, output_padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.block = Conv2dBlock(dec_channels + enc_channels, dec_channels, use_bn=use_bn)

    def forward(self, x: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder block.

        Args:
            x: Feature map from the previous decoder stage (to be upsampled).
            enc: Encoder feature map for skip connection.

        Returns:
            Output feature map after upsampling and refinement.
        """
        x = self.relu(self.conv_trans(x))
        x = torch.cat([x, enc], dim=1)
        return self.block(x)


class ConvNeXtPoseUNet(nn.Module):
    """
    Builds a U-Net-like architecture for landmark estimation using ConvNeXt-Base
    as the encoder backbone.

    Args:
        nkpts: Number of keypoints to predict, corresponding to heatmap channels.
        img_channels: Number of input image channels.
        filters: Base number of filters for decoder blocks.
        use_bn: Whether to apply Batch Normalization.
    """
    
    def __init__(
        self, 
        nkpts: int, 
        img_channels: int = 3, 
        filters: int = 64, 
        use_bn: bool = True
    ):
        super().__init__()

        # Encoder (contracting path)
        self.encoder = timm.create_model(
            "convnext_base",
            pretrained=True,
            features_only=True,
            out_indices=[0, 1, 2, 3]
        )

        # Decoder (expanding path) with skip connections
        self.dec4 = DecoderBlock(filters * 8, filters * 8, use_bn=use_bn)
        self.dec3 = DecoderBlock(filters * 4, filters * 4, use_bn=use_bn)
        self.dec2 = DecoderBlock(filters * 2, filters * 2, use_bn=use_bn)
        self.dec1 = DecoderBlock(filters, img_channels, stride=4, padding=0, use_bn=use_bn)

        # Output layer: generates one heatmap per keypoint
        self.output = nn.Sequential(
            nn.Conv2d(filters, nkpts, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        Args:
            x: Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            Predicted heatmaps of shape (batch_size, nkpts, height, width).
        """
        enc = self.encoder(x)

        dec4 = self.dec4(enc[3], enc[2])
        dec3 = self.dec3(dec4, enc[1])
        dec2 = self.dec2(dec3, enc[0])
        dec1 = self.dec1(dec2, x)

        return self.output(dec1)
    

def polynomial_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    decay_steps: int, 
    lr0: float, 
    lrf: float, 
    power: float
) -> LambdaLR:
    """
    Build a polynomial learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        decay_steps: Number of steps until reaching final LR.
        lr0: Initial learning rate.
        lrf: Final learning rate.
        power: Polynomial decay exponent.

    Returns:
        Polynomial LR scheduler.
    """
    def polynomial_decay(step):
        step = min(step, decay_steps)
        decayed_lr = (lr0 - lrf) * ((1 - (step / decay_steps)) ** power) + lrf
        return decayed_lr / lr0

    return LambdaLR(optimizer, lr_lambda=polynomial_decay)


def PoseLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Computes the total pose estimation loss as a combination of a L2 Soft Jaccard Loss
    for heatmap similarity and a Binary Cross Entropy loss for visibility prediction.

    Args:
        y_pred: Predicted heatmaps of shape (batch_size, nkpts, height, width).
        y_true: Ground truth heatmaps with the same shape as y_pred.

    Returns:
        A scalar representing the average loss over the batch.
    """
    # Compute Jaccard loss for each heatmap
    intersect_area = (y_true * y_pred).sum(dim=[2, 3])
    combined_area = (y_true.square() + y_pred.square()).sum(dim=[2, 3])
    iou_loss = 1 - divide_no_nan(intersect_area, combined_area - intersect_area)

    # Estimate keypoint visibility as maximum activation per heatmap
    vis_true = y_true.amax(dim=[2, 3])
    vis_pred = y_pred.amax(dim=[2, 3])

    # Weight Jaccard loss by visibility (only visible keypoints contribute)
    iou_loss = divide_no_nan((iou_loss * vis_true).sum(), vis_true.sum())

    # Compute binary cross entropy for visibility prediction
    bce_loss = nn.functional.binary_cross_entropy(vis_pred, vis_true)

    # Compute total loss
    loss = 8.0 * iou_loss + 0.1 * bce_loss
    return loss


class OKSMetric(Metric):
    """
    Computes the official COCO Object Keypoint Similarity (OKS) metric
    for evaluating keypoint predictions, assuming a single object per image.

    Args:
        sigmas: Standard deviations for each keypoint.
        name: Name of the metric.
    """
    
    def __init__(self, sigmas: list[float] | np.ndarray, name: str = "oks", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.register_buffer("sigmas", torch.tensor(sigmas, dtype=torch.float32))
        self.add_state("total_oks", torch.zeros((), dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros((), dtype=torch.float32), dist_reduce_fx="sum")

    def update(
        self, 
        kpts_pred: torch.Tensor, 
        kpts_true: torch.Tensor, 
        areas: torch.Tensor
    ) -> None:
        """
        Accumulates the OKS over a batch.

        Args:
            kpts_pred: Predicted keypoints of shape (batch_size, nkpts, 3).
            kpts_true: Ground-truth keypoints with the same shape as kpts_pred.
            areas: Object segmentation areas of shape (batch_size,).
        """
        # Compute squared Euclidean distance
        d2 = (kpts_true[..., :2] - kpts_pred[..., :2]).square().sum(dim=2) # (B, nkpts)

        # Compute visibility mask and denominator based on area and sigmas
        delta = (kpts_true[..., 2] > 0).float()
        den = 2 * areas[:, None] * (2 * self.sigmas).square() # (B, nkpts)

        # Compute Object Keypoint Similarity
        ks = divide_no_nan(-d2, den).exp()
        oks = divide_no_nan((ks * delta).sum(dim=1), delta.sum(dim=1)) # (B,)

        self.total_oks += oks.sum()
        self.total += oks.numel()

    def compute(self) -> torch.Tensor:
        """Computes the final mean OKS."""
        return divide_no_nan(self.total_oks, self.total)


class PCKMetric(Metric):
    """
    Computes the Percentage of Correct Keypoints (PCK) metric, normalized
    by the diagonal length of the object's bounding box. Only visible
    keypoints are considered. Assumes one object per image.

    Args:
        alpha: Threshold for considering a keypoint as correctly predicted.
        name: Name of the metric.
    """
    
    def __init__(self, alpha: float = 0.1, name: str = "pck", **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.add_state("total_correct", torch.zeros((), dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros((), dtype=torch.float32), dist_reduce_fx="sum")

    def update(
        self, 
        kpts_pred: torch.Tensor, 
        kpts_true: torch.Tensor, 
        diagonals: torch.Tensor
    ) -> None:
        """
        Accumulates the PCK over a batch.

        Args:
            kpts_pred: Predicted keypoints of shape (batch_size, nkpts, 3).
            kpts_true: Ground-truth keypoints with the same shape as kpts_pred.
            diagonals: Diagonals of the bounding boxes, shape (batch_size,).
        """
        # Compute Euclidean distance
        d = (kpts_true[..., :2] - kpts_pred[..., :2]).square().sum(dim=2).sqrt() # (B, nkpts)

        # Compute visibility mask and correct prediction mask
        visible_mask = (kpts_true[..., 2] > 0).float() # (B, nkpts)
        correct_mask = ((d / diagonals[:, None]) < self.alpha).float()

        self.total_correct += (correct_mask * visible_mask).sum()
        self.total += visible_mask.sum()

    def compute(self) -> torch.Tensor:
        """Computes the final mean PCK."""
        return divide_no_nan(self.total_correct, self.total)