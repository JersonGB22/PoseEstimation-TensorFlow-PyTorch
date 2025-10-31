import torch
import torchmetrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go


class PoseVisualizer():
    """
    Utility for visualizing pose estimation results: keypoints and skeleton connections.

    Args:
        skeleton: Maps each keypoint name to a two-element list containing its own ID 
            and its connection ID.
    """
    
    def __init__(self, skeleton: dict[str, list[int]]):
        self.skeleton = skeleton

    def draw_pose(
        self, 
        ax: plt.Axes, 
        keypoints: np.ndarray, 
        show_points: bool = True, 
        show_lines: bool = True, 
        show_kpts_ids: bool = True
    ) -> None:
        """
        Draws keypoints and skeleton on a given Matplotlib Axes.

        Args:
            ax: Matplotlib axes to draw on.
            keypoints: Array of shape (nkpts, 3) with (x, y, v).
            show_points: Whether to plot keypoint markers.
            show_lines: Whether to draw skeleton connections.
            show_kpts_ids: Whether to annotate keypoints with their IDs.
        """
        for ids in self.skeleton.values():
            # Draw line if both keypoints are visible
            if show_lines and np.all(keypoints[ids, 2] > 0):
                ax.plot(
                    keypoints[ids, 0],
                    keypoints[ids, 1],
                    color="blue",
                    linewidth=1.5,
                    zorder=1
                )

            # Draw individual keypoint if visible
            id = ids[0]
            x, y, v = keypoints[id]
            if v > 0:
                # Draw keypoint as a circle
                if show_points:
                    ax.scatter(
                        x, y,
                        facecolors="cyan",
                        edgecolors="blue",
                        linewidths=1.5,
                        s=(80 if id > 9 else 70) if show_kpts_ids else 50,
                        zorder=2
                    )

                # Display keypoint ID as label
                if show_kpts_ids:
                    ax.text(
                        x, y, str(id),
                        fontsize=5.25, fontweight="bold",
                        ha="center", va="center", zorder=3
                    )

    def plot_keypoints(
        self, 
        results: dict[str, np.ndarray], 
        metric: torchmetrics.Metric | None = None, 
        n_rows: int = 2, 
        random: bool = True, 
        show_points: bool = True,
        show_boxes: bool = True, 
        show_lines: bool = True, 
        show_kpts_ids: bool = True
    ) -> None:
        """
        Shows true vs predicted keypoints and optionally computes metrics on each sample.

        Args:
            results: Dict expected to contain images, bboxes, keypoints, 
                and keypoints_pred as numpy arrays.
            metric: Metric instance to update and display per sample.
            n_rows: Number of rows.
            random: Shuffle sample order if True.
            show_points: Whether to plot keypoint markers.
            show_boxes: Whether to draw bounding boxes.
            show_lines: Whether to draw skeleton connections.
            show_kpts_ids: Whether to annotate keypoints with their IDs.
        """
        indices = np.arange(len(results["images"]))
        if random:
            np.random.shuffle(indices)

        fig, axes = plt.subplots(n_rows, 2, figsize=(9.6, 4.8 * n_rows))
        axes = np.array(axes).reshape(n_rows, 2)

        for col, title in enumerate(["True Keypoints", "Pred Keypoints"]):
            axes[0, col].set_title(title, fontsize=9)

        for i in range(n_rows):
            idx = indices[i]
            image = results["images"][idx]
            kpts_true = results["keypoints"][idx]
            kpts_pred = results["keypoints_pred"][idx]
            if show_boxes or (metric is not None):
                bbox = results["bboxes"][idx]

            ax = axes[i, 0]
            ax.imshow(image)
            ax.axis("off")
            self.draw_pose(ax, kpts_true, show_points, show_lines, show_kpts_ids)

            ax = axes[i, 1]
            ax.imshow(image)
            ax.axis("off")
            self.draw_pose(ax, kpts_pred, show_points, show_lines, show_kpts_ids)

            if metric is not None:
                if metric.name == "oks":
                    normalizer = np.prod(bbox[None, 2:], axis=1) * 0.53
                elif metric.name == "pck":
                    normalizer = np.linalg.norm(bbox[None, 2:], axis=1)
                else:
                    raise ValueError("Accepted metrics are OKS and PCK.")

                metric.reset()
                metric.update(
                    torch.tensor(kpts_pred[None, :], dtype=torch.float32, device=metric.device),
                    torch.tensor(kpts_true[None, :], dtype=torch.float32, device=metric.device),
                    torch.tensor(normalizer, dtype=torch.float32, device=metric.device),
                )

                shape = image.shape
                ax.text(
                    x=shape[1] / 2, y=shape[0],
                    s=f"{metric.name.upper()}: {metric.compute().item():.3f}",
                    fontsize=5.41, fontweight="bold", ha="center", va="top",
                )

            if show_boxes:
                rect1 = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1.75, edgecolor="red", facecolor="none")
                rect2 = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1.75, edgecolor="red", facecolor="none")
                axes[i, 0].add_patch(rect1)
                axes[i, 1].add_patch(rect2)

        plt.tight_layout()
        plt.show()


def plot_training_curves(
    history: pd.DataFrame, 
    metrics: list[str] | None = None, 
    idxmax: int | None = None, 
    renderer: str | None = None
) -> None:
    """
    Plots the evolution of training, validation and/or testing metrics over epochs using Plotly.

    Args:
        history: DataFrame containing the training history with one column per metric.
        metrics: List of metric names to plot. If None, all columns will be plotted.
        idxmax: Index of the best epoch based on the monitored metric. 
            If None, no vertical line is shown.
        renderer: Renderer to use for Plotly (e.g., 'notebook', 'colab', 'png').
    """
    epochs = len(history)
    if metrics is None:
        metrics = history.columns.tolist()

    list_epochs = np.arange(1, epochs + 1)
    fig = go.Figure()

    for metric in metrics:
        if metric not in history.columns:
            raise ValueError(f"Metric '{metric}' not found in history columns.")

        # Add line plot for the current metric
        fig.add_trace(go.Scatter(
            x=list_epochs,
            y=history[metric],
            name=metric,
            mode="lines",
            line=dict(width=2)
        ))

    # Add vertical line for early stopping point, if provided
    if idxmax is not None:
        fig.add_vline(
            x=idxmax + 1,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text="Early Stopping",
            annotation_position="top left",
            annotation=dict(font_size=12, font_color="red")
        )

    # Determine whether we're plotting losses, metrics, or mixed
    is_loss = all("loss" in m.lower() for m in metrics)
    is_metric = all("loss" not in m.lower() for m in metrics)

    title_text = "Loss Evolution" if is_loss else (
        "Metric Evolution" if is_metric else "Loss and Metric Evolution"
    )
    yaxis_text = "Loss" if is_loss else (
        "Metric" if is_metric else "Value"
    )

    # Axis and layout customization
    fig.update_layout(
        title=title_text,
        title_font=dict(size=18),
        title_x=0.5,
        xaxis_title="Epoch",
        yaxis_title=yaxis_text,
        width=1000,
        height=500,
        showlegend=True
    )

    fig.show(renderer=renderer)