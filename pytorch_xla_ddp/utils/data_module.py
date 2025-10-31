import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np
from PIL import Image
from glob import glob
import argparse
import cv2
import os


def divide_no_nan(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    """Avoid division by zero."""
    return torch.where(den != 0, num / den, 0)


class HorizontalFlipSymmetricKeypoints():
    """
    Randomly applies a horizontal flip to an image and its keypoints,
    taking into account symmetric keypoints and their visibility.

    Args:
        symmetric_kpts: List of pairs of symmetric keypoint indices.
        p: Probability of applying the horizontal flip.
    """
    
    def __init__(self, symmetric_kpts: list[list[int]] | None = None, p: float = 0.5):
        self.symmetric_kpts = symmetric_kpts
        self.p = p

    def __call__(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies a horizontal flip to the image and its keypoints.

        Args:
            image: Image array of shape (height, width, 3).
            keypoints: Array of keypoints with shape (nkpts, 3),
                where each keypoint is (x, y, visibility).

        Returns:
            Flipped image and transformed keypoints, with symmetric pairs swapped.
        """
        if np.random.rand() >= self.p:
            return image, keypoints

        # Apply standard horizontal flip to the image
        image_flip = np.ascontiguousarray(np.fliplr(image))
        keypoints_flip = keypoints.copy()
        width = image.shape[1]

        # Flip x-coordinates of visible keypoints
        mask = keypoints_flip[:, 2] > 0
        keypoints_flip[mask, 0] = np.clip(width - keypoints_flip[mask, 0], 0, width - 1e-6)

        # Swap symmetric keypoint pairs
        if self.symmetric_kpts:
            for pair in self.symmetric_kpts:
                keypoints_flip[pair] = keypoints_flip[pair[::-1]]

        return image_flip, keypoints_flip


def keypoints_to_heatmaps(
    keypoints: np.ndarray, 
    heatmap_size: tuple[int, int], 
    kernel_size: tuple[int, int] = (33, 33), 
    sigma: float = 6.0
) -> np.ndarray:
    """
    Converts keypoints into heatmaps using Gaussian blur.

    Args:
        keypoints: Array of shape (nkpts, 3), where each row is (x, y, visibility).
        heatmap_size: Heatmap size as (height, width).
        kernel_size: Size of the Gaussian kernel (must be odd numbers).
        sigma: Standard deviation used for Gaussian blur.

    Returns:
        Array of heatmaps with shape (nkpts, height, width).
    """
    heatmaps = []
    for x, y, v in keypoints.astype(np.int32):
        # Initialize an empty heatmap for the current keypoint
        heatmap = np.zeros(heatmap_size, np.float32)

        # Apply Gaussian only if the keypoint is labeled
        if v > 0:
            heatmap[y, x] = 1
            heatmap = cv2.GaussianBlur(heatmap, kernel_size, sigma, borderType=cv2.BORDER_CONSTANT)
            heatmap /= heatmap.max() # Normalize heatmap to [0, 1] range

        heatmaps.append(heatmap)
    return np.stack(heatmaps)


def heatmaps_to_keypoints(heatmaps: torch.Tensor, mode: str = "max") -> torch.Tensor:
    """
    Extracts (x, y, visibility) keypoint coordinates from heatmaps.
    Two methods are available:
        - "max": takes the position of the maximum value in the heatmap.
        - "mean": computes the center of mass (soft-argmax) as a weighted average.

    Args:
        heatmaps: Tensor of shape (batch_size, nkpts, height, width).
        mode: One of {"max", "mean"} indicating the extraction method.

    Returns:
        Keypoints of shape (batch_size, nkpts, 3), where each row is (x, y, v).
            Visibility is set to 2 for detected keypoints (confidence > 0.5),
            and 0 for undetected ones, following the COCO format.
    """
    if mode not in {"max", "mean"}:
        raise ValueError("Invalid method. Use 'max' or 'mean'.")

    B, N, H, W = heatmaps.shape
    device = heatmaps.device

    # Flatten spatial dimensions for easier indexing
    heatmaps_flat = heatmaps.view(B, N, -1)

    if mode == "max":
        # Get index of the maximum activation for each heatmap
        indices = heatmaps_flat.argmax(dim=-1)
        x, y = indices % W, indices // W

    else:
        # Normalize heatmaps to form valid probability distributions
        heatmaps = divide_no_nan(heatmaps, heatmaps.sum(dim=[2, 3], keepdim=True))

        # Generate x and y coordinate grids
        xs = torch.arange(W, device=device).view(1, 1, 1, W)
        ys = torch.arange(H, device=device).view(1, 1, H, 1)

        # Compute weighted averages along each axis
        x = (heatmaps * xs).sum(dim=[2, 3])
        y = (heatmaps * ys).sum(dim=[2, 3])

    # Compute keypoints and set visibility to 2 if confidence > 0.5, otherwise 0
    v = (heatmaps_flat.amax(dim=-1) > 0.5).to(x.dtype)
    keypoints = torch.stack([x, y, 2 * v], dim=-1)

    return keypoints.float()


class PoseDataset(Dataset):
    """
    Custom dataset for pose estimation tasks.

    Args:
        args: Parsed configuration containing general parameters for training, 
            evaluation, and inference.
        split: Either 'train' or 'test', indicating dataset partition.
        training: If True, applies data augmentation and generates images, keypoints, 
            and heatmaps. If False, skips augmentation and also includes bounding boxes.
    """
    
    def __init__(
        self, 
        args: argparse.Namespace, 
        split: str = "train", 
        training: bool = True
    ):
        if split not in {"train", "test"}:
            raise ValueError("Invalid dataset split. Must be 'train' or 'test'.")

        # Load image and label paths based on the split
        self.image_paths = sorted(glob(os.path.join(args.data_dir, f"{split}/images/*.jpg")))
        self.label_paths = sorted(glob(os.path.join(args.data_dir, f"{split}/labels/*.txt")))

        self.height, self.width = args.img_size
        self.args = args
        self.training = training

        # Define data augmentation transforms used during training
        if training:
            self.random_horizontal_flip = HorizontalFlipSymmetricKeypoints(args.symmetric_kpts, p=0.5)

            self.random_affine = A.Compose(
                [
                    A.Affine(
                    scale=(0.7, 1.3),
                    translate_percent=(-0.2, 0.2),
                    rotate=(-30, 30),
                    shear=(0.0, 0.0),
                    p=0.6
                )
                ],
                keypoint_params=A.KeypointParams(
                    format="xy",
                    remove_invisible=False,
                    check_each_transform=True
                ),
                p=1.0
            )

            self.random_brightness_contrast = A.Compose(
                [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5)],
                p=1.0
            )

    def data_augmentation(
        self, 
        image: np.ndarray, 
        keypoints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to image and keypoints.

        Args:
            image: Input image as (height, width, 3).
            keypoints: Array of keypoints (nkpts, 3) in (x, y, visibility) format.

        Returns:
            Augmented image and keypoints.
        """
        image, keypoints = self.random_horizontal_flip(image, keypoints)

        transformed = self.random_affine(image=image, keypoints=keypoints[:, :2])
        keypoints_trans = transformed["keypoints"]

        # Create mask for valid keypoints after affine transformation
        mask = (keypoints[:, 2] > 0) & (
            (keypoints_trans[:, 0] >= 0) & (keypoints_trans[:, 1] >= 0) &
            (keypoints_trans[:, 0] < self.width) & (keypoints_trans[:, 1] < self.height)
        )

        # Apply the affine transformation only if at least one keypoint is visible
        if np.sum(mask) > 0:
            keypoints_trans = np.concatenate([keypoints_trans, keypoints[:, 2:]], axis=-1)
            keypoints = np.where(mask[:, None], keypoints_trans, np.zeros_like(keypoints_trans))
            image = transformed["image"]

        image = self.random_brightness_contrast(image=image)["image"]
        return image, keypoints

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Loads, processes, and returns a single sample from the dataset by index.
        """
        # Load and resize image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image.resize((self.width, self.height)))

        # Load label: [xmin, ymin, width, height, x1, y1, v1, ..., xn, yn, vn]
        label = np.loadtxt(self.label_paths[idx], dtype=np.float32)

        # Extract and denormalize keypoints
        keypoints = label[4:].reshape(-1, 3)
        keypoints[:, 0] *= self.width
        keypoints[:, 1] *= self.height

        # Apply augmentation if training
        if self.training:
            image, keypoints = self.data_augmentation(image, keypoints)

        # Generate heatmaps from keypoints using Gaussian blur
        keypoints = keypoints.astype(np.int32)
        heatmaps = keypoints_to_heatmaps(
            keypoints, (self.height, self.width),
            self.args.kernel_size, self.args.heatmap_sigma
        )

        # Convert keypoints and heatmaps to tensors
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        # Convert image to tensor and normalize
        image = torch.tensor(image, dtype=torch.float32)
        image = ((image / 255) - self.args.img_mean) / self.args.img_std
        image = image.permute(2, 0, 1)

        data = {"images": image, "heatmaps": heatmaps, "keypoints": keypoints}

        # Extract and denormalize bounding box if configured
        if not self.training:
            data["bboxes"] = torch.tensor([
                label[0] * self.width,
                label[1] * self.height,
                label[2] * self.width,
                label[3] * self.height
            ], dtype=torch.float32)

        return data