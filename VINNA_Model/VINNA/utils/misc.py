import os
from itertools import product
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision import utils
from skimage import color
try:
    import wandb
except Exception as e:
    print(f"Wandb not installed: {e}")

from VINNA.utils import colormapper as col


def log_image_table(images, predicted, labels, probs, num_classes=34):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(num_classes)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
    wandb.log({"predictions_table":table}, commit=False)


def prepare_label_grid(img, nrow=4, bglabel=0, mid_depth=64, colormap=None):
    """
    Make label grid from image with torch.utils.make_grid.
    Tensor shape must be B x C x H x W.
    Normally, labels and batch output have B x H x W (x D for 3D).
    :param img:
    :param nrow:
    :param bglabel:
    :return:
    """
    if img.shape[1] != 1 and len(img.shape) == 3:
        # B x H x W to B x C x H x W
        img = img.unsqueeze_(1)
    elif img.shape[1] != 1 and len(img.shape) == 4:
        # B x H x W x D to B x C x H x W
        # --> batch size is one, so transpose to use depth as batch (D x C x H x W)
        img = torch.transpose(img, 3, 0)[mid_depth - 4: mid_depth + 5, ..., 0].unsqueeze_(1)
    elif len(img.shape) == 5:
        # B x C x H x W x D to B x C x H x W
        # --> batch size is one, so transpose to use depth as batch (D x C x H x W)
        img = torch.transpose(img, 4, 0)[mid_depth - 4: mid_depth + 5, ..., 0]
    assert len(img.shape) == 4
    lgrid = utils.make_grid(img.cpu(), nrow=nrow)[0]
    if colormap is None:
        mapped_labels = color.label2rgb(lgrid.numpy(), bg_label=bglabel, colors=colormap)
    else:
        mapped_labels = col.map_grid_to_colormap(colormap, lgrid)
    return mapped_labels


def plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name, wandb, wandb_log, fs_colors=None, limit=8):
    """
    Function to plot predictions from validation set.
    :param torch.Tensor images_batch: input images
    :param torch.Tensor labels_batch: segmentation ground truth
    :param torch.Tensor batch_output: network prediction
    :param str plt_title: title of the plot
    :param str file_save_name: path and filename to save image as. Can be empty to show image in line
    :param wandb.Log wandb: wandb instance to write image to
    :param str wandb_log: wandb log-dir
    :param list fs_colors: color table to apply for segmentations and network predictions
    :param int limit: reduce batch size to this limit to avoid overcrowding of the resulting plot
                      (default=8, disable with setting to 0)
    :return None: saving image or displaying it directly with plt.show()
    """
    if limit > 0:
        images_batch, labels_batch, batch_output = images_batch[0:limit, ...], \
                                                   labels_batch[0:limit, ...], \
                                                   batch_output[0:limit, ...]

    if images_batch.shape[-1] == 7:
        # Channel is last dim, transpose to second (0, 1, 2, 3) --> (0, 3, 1, 2)
        images_batch, labels_batch, batch_output = torch.transpose(images_batch, -1, 1), \
                                                   torch.transpose(labels_batch, -1, 1), \
                                                   torch.transpose(batch_output, -1, 1)
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    mid_slice = images_batch.shape[1] // 2
    mid_depth=64
    if len(images_batch.shape) == 5:
        # 3D --> batch size is one, so transpose to use depth as batch (B x C x H x W x D --> D x C x H x W)
        # B x C x H x W x D --> D x C x H x W x B --> Crop x 1 x H x W x 1
        mid_depth = images_batch.shape[4] // 2
        images_batch = torch.transpose(images_batch, 4, 0)
        images_batch = images_batch[mid_depth - 4: mid_depth + 5, mid_slice, :, :, 0]
        images_batch = images_batch.unsqueeze_(1)

    else:
        # 2D (B x C x H x W)
        images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)

    # Image grid
    rows = int(np.sqrt(images_batch.shape[0]))
    grid = utils.make_grid(images_batch.cpu(), nrow=rows)
    ax3.imshow(grid.numpy().transpose((1, 2, 0)))

    # Label grid
    color_grid = prepare_label_grid(labels_batch, nrow=rows, mid_depth=mid_depth, colormap=fs_colors)
    ax1.imshow(grid.numpy().transpose((1, 2, 0)))
    ax1.imshow(color_grid, alpha=0.6)

    # Prediction grid
    color_grid = prepare_label_grid(batch_output, nrow=rows, mid_depth=mid_depth, colormap=fs_colors)
    ax2.imshow(grid.numpy().transpose((1, 2, 0)))
    ax2.imshow(color_grid, alpha=0.6)

    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    f.suptitle(plt_title)
    plt.tight_layout()
    if file_save_name is not None:
        f.savefig(file_save_name, bbox_inches='tight')
    else:
        plt.show()
    if wandb is not None:
        wandb.log({f"{wandb_log}/prediction_plots": f})
    plt.close(f)
    plt.gcf().clear()


def plot_confusion_matrix(cm,
                          classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(35, 35),
                          file_save_name="temp.pdf"):
    n_classes = len(classes)

    fig, ax = plt.subplots(figsize=figsize)
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None
    ax.set_title(title)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    fig.colorbar(im_, cax=cax)

    tick_marks = np.arange(n_classes)
    ax.set(xticks=tick_marks,
           yticks=tick_marks,
           xticklabels=classes,
           yticklabels=classes,
           ylabel="True label",
           xlabel="Predicted label")

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)

    values_format = '.2f'
    thresh = (cm.max() + cm.min()) / 2.0

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        text_[i, j] = ax.text(j, i,
                              format(cm[i, j], values_format),
                              ha="center", va="center",
                              color=color)

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='horizontal')

    return fig


def find_latest_experiment(path):
    list_of_experiments = os.listdir(path)
    list_of_int_experiments = []
    for exp in list_of_experiments:
        try:
            int_exp = int(exp)
        except ValueError:
            continue
        list_of_int_experiments.append(int_exp)

    if len(list_of_int_experiments) == 0:
        return 0

    return max(list_of_int_experiments)


def check_path(path):
    os.makedirs(path, exist_ok=True)
    return path


def update_num_steps(dataloader, cfg):
    cfg.TRAIN.NUM_STEPS = len(dataloader)

