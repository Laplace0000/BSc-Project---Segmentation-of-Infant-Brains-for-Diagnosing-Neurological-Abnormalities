# IMPORTS
import pprint

import torch
import numpy as np
from tqdm import tqdm
import math
import time
import os
import pandas as pd
from collections import defaultdict
import wandb

from VINNA.models.networks import build_model
from VINNA.models.optimizer import get_optimizer
from VINNA.models.losses import get_loss_func

import VINNA.data_processing.data_loader.loader as loader
from VINNA.data_processing.utils.data_utils import read_classes_from_lut

import VINNA.utils.checkpoint as cp
from VINNA.utils.lr_scheduler import get_lr_scheduler
from VINNA.utils.metrics import iou_score, precision_recall, evaluate_metrics
from VINNA.utils.misc import update_num_steps, plot_predictions
import VINNA.utils.logging as logging_u

logger = logging_u.get_logger(__name__)


class Trainer:
    def __init__(self, cfg):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg
        self.mix = cfg.DATA.MIX_T1_T2
        # Create the checkpoint dir.
        self.checkpoint_dir = cp.create_checkpoint_dir(cfg.LOG_DIR, cfg.EXPR_NUM,
                                                       net=cfg.MODEL.MODEL_NAME, aug=cfg.DATA.AUGNAME)
        logging_u.setup_logging(cfg.LOG_DIR, cfg.EXPR_NUM)
        logger.info("Training with config:")
        logger.info(pprint.pformat(cfg))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}, cuda available: {torch.cuda.is_available()}")
        assert torch.cuda.is_available() == True
        self.model = build_model(cfg, train=True, device=self.device)
        self.loss_func = get_loss_func(cfg)
        self.n_steps_per_epoch = 0
        self.example_ct = 0

        # Define image type to load from hdf5
        self.imgt = cfg.DATA.IMG_TYPE

        # set up class names
        self.lut = read_classes_from_lut(cfg.DATA.LUT)
        self.class_names = self.lut["LabelName"].values[1:]
        self.fs_colors = self.lut[["R", "G", "B"]].values / 255
        if cfg.DATA.COMBINED_LABELS:
            if self.cfg.DATA.PLANE != "sagittal":
                self.combined_labels = {89: [6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38],  # Right-GM
                                    90: [5, 7, 9, 11, 13, 15, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39],  # Left-GM
                                    91: [51, 53, 55, 57, 59, 61, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82],  # Right-WM,
                                    92: [52, 54, 56, 58, 60, 62, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81],  # Left-WM
                                    93: [42, 86], 94: [43, 87]}  # Right- and Left-Thalamus
            else:
                self.combined_labels = {48: [3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  # Right-GM
                                        49: [48],  # Left-GM
                                        50: [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42],
                                        # Right-WM,
                                        51: [50],  # Left-WM
                                        52: [22, 46], 53: [52]}  # Right- and Left-Thalamus
            self.combined_names = ["Right-Cerebral-Cortex", "Left-Cerebral-Cortex",
                                   "Right-Cerebral-White-Matter", "Left-Cerebral-White-Matter",
                                   "Right-Thalamus", "Left-Thalamus"]

        # Meta info with number of slices for validation set
        self.meta = pd.read_csv(cfg.DATA.META_INFO, sep="\t")
        self.meta["SubjectWithRes"] = self.meta["SubjectFix"] + "+" + self.meta["Resolution"].astype(str)

        # Set up logger format
        self.a = "{}\t" * (cfg.MODEL.NUM_CLASSES - 2) + "{}"
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.plot_dir = os.path.join(cfg.LOG_DIR, "pred", cfg.MODEL.MODEL_NAME, cfg.DATA.AUGNAME, str(cfg.EXPR_NUM))
        os.makedirs(self.plot_dir, exist_ok=True)

        self.project = cfg.WANDB_PROJECT
        self.subepoch = False if self.cfg.TRAIN.BATCH_SIZE == 16 else True

        # Model set up
        if self.cfg.NUM_GPUS > 1:
            assert self.cfg.NUM_GPUS <= torch.cuda.device_count(), \
                "Cannot use more GPU devices than available"
            print("Using ", self.cfg.NUM_GPUS, "GPUs!")
            self.model = torch.nn.DataParallel(self.model)

        # Transfer the model to device(s)
        self.model = self.model.to(self.device)

    def train(self, train_loader, optimizer, scheduler, wandb, epoch):
        self.model.train()
        logger.info("Training started ")
        epoch_start = time.time()
        loss_batch = np.zeros(1)

        for curr_iter, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Typecast labels to long tensor --> labels are bytes initially (uint8),
            # index operations requiere LongTensor in pytorch
            images, labels, weights, scale_factors, affine, inverse = batch[self.imgt].to(self.device), \
                                                              batch['label'].type(torch.LongTensor).to(self.device), \
                                                                         batch['weight'].float().to(self.device), \
                                                                         batch['scale_factor'].to(self.device), \
                                                                         batch['affine'].to(self.device), \
                                                                         batch['inverse'].to(self.device)

            # Assert neither image nor label has nan --> issues with loss
            assert not torch.any(torch.isnan(images))
            assert not torch.any(torch.isnan(labels))
            assert not torch.any(torch.isnan(weights))

            if not self.subepoch or curr_iter % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.zero_grad()  # every second epoch to get batchsize of 16 if using 8

            pred = self.model(images, scale_factors, affine, inverse)

            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)
            assert not torch.any(torch.isnan(loss_total))
            loss_total.backward()
            loss_batch += loss_total.item()
            assert not np.any(np.isnan(loss_batch))
            if not self.subepoch or (curr_iter + 1) % (16 / self.cfg.TRAIN.BATCH_SIZE) == 0:
                optimizer.step()  # every second epoch to get batchsize of 16 if using 8
                if scheduler is not None and self.cfg.OPTIMIZER.LR_SCHEDULER != "reduceLROnPlateau":
                    scheduler.step(epoch + curr_iter / len(train_loader))

            self.example_ct += len(images)

            # Log metrics to wandb
            metrics = {"train/train_loss_total": loss_total,
                       "train/train_loss_dice": loss_dice,
                       "train/train_loss_ce": loss_ce,
                       "train/epoch": epoch,
                       "train/example_ct": self.example_ct,
                       "train/learning_rate": optimizer.param_groups[0]['lr']}

            # 🐝 Log train metrics to wandb
            #wandb.log(metrics)
            # Plot sample predictions
            if curr_iter == len(train_loader) - 2 and ((epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0 or epoch == 0):
                plt_title = 'Training Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                            'Epoch_' + str(epoch) + '_Training_Predictions.png')

                _, batch_output = torch.max(pred, dim=1)
                if self.cfg.DATA.DIMENSION == 3:
                    images = self.model.crop_vol_to_patch(images)
                plot_predictions(images, labels, batch_output, plt_title, file_save_name, wandb, "train",
                                 self.fs_colors)

            if curr_iter % (len(train_loader) // 2) == 0 or curr_iter == len(train_loader) - 1:
                logger.info("Train Epoch: {} [{}/{}] ({:.0f}%)] "
                            "with loss: {}".format(epoch, curr_iter,
                                                   len(train_loader),
                                                   100. * curr_iter / len(train_loader),
                                                   loss_batch / (curr_iter + 1)))

        logger.info("Training epoch {} finished in {:.04f} seconds".format(epoch, time.time() - epoch_start))

    @torch.no_grad()
    def eval(self, val_loader, wandb, epoch):
        logger.info(f"Evaluating model at epoch {epoch}")
        self.model.eval()

        val_loss_total = defaultdict(float)
        val_loss_dice = defaultdict(float)
        val_loss_ce = defaultdict(float)

        ints_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        unis_ = defaultdict(lambda: np.zeros(self.num_classes - 1))
        miou = np.zeros(self.num_classes - 1)
        per_cls_counts_gt = defaultdict(lambda: np.zeros(self.num_classes - 1))
        per_cls_counts_pred = defaultdict(lambda: np.zeros(self.num_classes - 1))
        accs = defaultdict(
            lambda: np.zeros(self.num_classes - 1))  # -1 to exclude background (still included in val loss)

        val_start = time.time()
        for curr_iter, batch in tqdm(enumerate(val_loader), total=len(val_loader)):

            labels, weights, scale_factors, affine, inverse = batch['label'].type(torch.LongTensor).to(
                self.device), \
                                                                             batch['weight'].float().to(self.device), \
                                                                             batch['scale_factor'].to(self.device), \
                                                                             batch["affine"].to(self.device), \
                                                                             batch["inverse"].to(self.device)

            if self.mix:
                # Mix batches with T1 and T2 images
                images, images_t2 = batch["image"], batch["t2_image"]
                images[1::2, ...] = images_t2[1::2, ...]
                images = images.to(self.device)
            else:
                images = batch[self.imgt].to(self.device)

            pred = self.model(images, scale_factors, affine, inverse)
            loss_total, loss_dice, loss_ce = self.loss_func(pred, labels, weights)

            sf = torch.unique(scale_factors)
            if len(sf) == 1:
                sf = sf.item()
                val_loss_total[sf] += loss_total.item()
                val_loss_dice[sf] += loss_dice.item()
                val_loss_ce[sf] += loss_ce.item()

                _, batch_output = torch.max(pred, dim=1)

                # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                int_, uni_ = iou_score(batch_output, labels, self.num_classes)
                ints_[sf] += int_
                unis_[sf] += uni_

                tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels, self.num_classes)
                accs[sf] += tpos
                per_cls_counts_gt[sf] += pcc_gt
                per_cls_counts_pred[sf] += pcc_pred

            # Plot sample predictions
            if curr_iter == (len(val_loader) // 3) and (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                # log_image_table(images, batch_output, labels, pred.softmax(dim=1))
                plt_title = 'Validation Results Epoch ' + str(epoch)

                file_save_name = os.path.join(self.plot_dir,
                                              'Epoch_' + str(epoch) + '_Validations_Predictions.png')

                plot_predictions(images, labels, batch_output, plt_title, file_save_name, wandb, "val", self.fs_colors)

        logger.info("Validation epoch {} finished in {:.04f} seconds".format(epoch, time.time() - val_start))

        # Get final measures and log them
        if len(accs.keys()) > 3:
            logger.info("Something is wrong, to many keys detected: {}".format(len(accs.keys())))
            logger.info(accs.keys())
            exit()
        for key in accs.keys():
            ious = ints_[key] / unis_[key]
            miou += ious
            val_loss_total[key] /= (curr_iter + 1)
            val_loss_dice[key] /= (curr_iter + 1)
            val_loss_ce[key] /= (curr_iter + 1)

            val_metrics = {f"val/{sf}/val_loss_total": val_loss_total[sf],
                           f"val/{sf}/val_loss_dice": val_loss_dice[sf],
                           f"val/{sf}/val_loss_ce": val_loss_ce[sf],
                           # f"val/{sf}/val_accuracy": accs[key] / (per_cls_counts_gt[sf] + per_cls_counts_pred[sf]),
                           f"val/{sf}/val_miou": np.nanmean(ious),
                           f"val/{sf}/val_recall": np.nanmean(accs[key] / per_cls_counts_gt[key]),
                           f"val/{sf}/val_precision": np.nanmean(accs[key] / per_cls_counts_pred[key]),

                           }
            #wandb.log({**val_metrics})

            # Log metrics
            logger.info("[Epoch {} stats]: SF: {}, MIoU: {:.4f}; "
                        "Mean Recall: {:.4f}; "
                        "Mean Precision: {:.4f}; "
                        "Avg loss total: {:.4f}; "
                        "Avg loss dice: {:.4f}; "
                        "Avg loss ce: {:.4f}".format(epoch, key, np.nanmean(ious),
                                                     np.nanmean(accs[key] / per_cls_counts_gt[key]),
                                                     np.nanmean(accs[key] / per_cls_counts_pred[key]),
                                                     val_loss_total[key], val_loss_dice[key], val_loss_ce[key]))

            logger.info(self.a.format(*self.class_names))
            logger.info(self.a.format(*ious))

        return np.nanmean(np.nanmean(miou))

    def run(self):

        val_loader = loader.get_dataloader(self.cfg, "val")
        train_loader = loader.get_dataloader(self.cfg, "train")

        update_num_steps(train_loader, self.cfg)
        self.n_steps_per_epoch = math.ceil(len(train_loader.dataset) / self.cfg.TRAIN.BATCH_SIZE)
        optimizer = get_optimizer(self.model, self.cfg)
        scheduler = get_lr_scheduler(optimizer, self.cfg)
        checkpoint_paths = cp.get_checkpoint_path(self.cfg.LOG_DIR,
                                                  self.cfg.TRAIN.RESUME_EXPR_NUM,
                                                  net=self.cfg.MODEL.MODEL_NAME,
                                                  aug=self.cfg.DATA.AUGNAME)
        if self.cfg.TRAIN.RESUME and checkpoint_paths:
            try:
                checkpoint_path = checkpoint_paths.pop()
                checkpoint_epoch, best_metric = cp.load_from_checkpoint(
                    checkpoint_path,
                    self.model,
                    optimizer,
                    scheduler,
                    self.cfg.TRAIN.FINE_TUNE
                )
                start_epoch = checkpoint_epoch
                best_miou = best_metric
                logger.info(f"Resume training from epoch {start_epoch}")
            except Exception as e:
                print("No model to restore. Resuming training from Epoch 0. {}".format(e))
        else:
            logger.info("Training from scratch")
            start_epoch = 0
            best_miou = 0

        logger.info("{} parameters in total".format(sum(x.numel() for x in self.model.parameters())))
        # Create wandb experiment
        """
        if start_epoch <= self.cfg.TRAIN.NUM_EPOCHS:
            wandb.init(
                # Set the project where this run will be logged
                project=f"{self.project}",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=f"{self.cfg.EXPR_NUM}",
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": self.cfg.OPTIMIZER.BASE_LR,
                    "architecture": self.cfg.MODEL.MODEL_NAME,
                    "batch_size": self.cfg.TRAIN.BATCH_SIZE,
                    "augmentations": self.cfg.DATA.AUG,
                    "lr_scheduler": self.cfg.OPTIMIZER.LR_SCHEDULER,
                    "optimizer": self.cfg.OPTIMIZER.OPTIMIZING_METHOD,
                    "dataset": "dHCP",
                    "epochs": self.cfg.TRAIN.NUM_EPOCHS,
                })
        """
        wandb=None
        logger.info("Summary path {}".format(self.cfg.SUMMARY_PATH))
        # Perform the training loop.
        logger.info("Start epoch: {}".format(start_epoch + 1))

        for epoch in range(start_epoch, self.cfg.TRAIN.NUM_EPOCHS):
            self.train(train_loader, optimizer, scheduler, wandb, epoch=epoch)

            miou = self.eval(val_loader, wandb, epoch=epoch)

            if self.cfg.OPTIMIZER.LR_SCHEDULER == "reduceLROnPlateau":
                scheduler.step(miou)

            if (epoch + 1) % self.cfg.TRAIN.CHECKPOINT_PERIOD == 0:
                logger.info(f"Saving checkpoint at epoch {epoch + 1}")
                cp.save_checkpoint(self.checkpoint_dir,
                                   epoch + 1,
                                   best_miou,
                                   self.cfg.NUM_GPUS,
                                   self.cfg,
                                   self.model,
                                   optimizer,
                                   scheduler
                                   )

            if miou > best_miou:
                best_miou = miou
                logger.info(
                    f"New best checkpoint reached at epoch {epoch + 1} with miou of {best_miou}\nSaving new best model.")
                cp.save_checkpoint(self.checkpoint_dir,
                                   epoch + 1,
                                   best_miou,
                                   self.cfg.NUM_GPUS,
                                   self.cfg,
                                   self.model,
                                   optimizer,
                                   scheduler,
                                   best=True
                                   )

        #if start_epoch <= self.cfg.TRAIN.NUM_EPOCHS:
            #wandb.finish()
