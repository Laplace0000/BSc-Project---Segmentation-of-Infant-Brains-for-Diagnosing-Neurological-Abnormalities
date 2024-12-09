import pprint
import torch
import numpy as np
import time
import yacs.config
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Union

import VINNA.utils.logging as logging_u
from VINNA.models.networks import build_model
from VINNA.data_processing.augmentation.augmentation import ToTensor, ZeroPad
from VINNA.data_processing.utils.data_utils import map_prediction_sagittal2full
from VINNA.data_processing.data_loader.dataset import MultiScaleOrigDataThickSlices


logger = logging_u.getLogger(__name__)


class Inference:
    """
    Model evaluation class to run inference using FastSurferCNN.

        Attributes
        ----------
        device : Optional[torch.device])
            Device specification for distributed computation usage.
        default_device : torch.device
            Default device specification for distributed computation usage.
        cfg : yacs.config.CfgNode
            Configuration Node

        Methods
        -------
        setup_model
            Set up the initial model
        set_cfg
            Set configuration node
        to
            Moves and/or casts the parameters and buffers.
        load_checkpoint
            Load the checkpoint
        eval
            Evaluate predictions
        run
            Run the loaded model
    """
    def __init__(self, cfg: yacs.config.CfgNode,
                 ckpt: str,
                 device: torch.device,
                 lut: Union[None, str, np.ndarray] = None):
        # Set random seed from configs.
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        self.cfg = cfg

        # Switch on denormal flushing for faster CPU processing
        # seems to have less of an effect on VINN than old CNN
        torch.set_flush_denormal(True)

        # Set up logging
        #logging_u.setup_logging(cfg.OUT_LOG_DIR, cfg.OUT_LOG_NAME)
        #logger.info("Run Inference with config:")
        #logger.info(pprint.pformat(cfg))

        # Define device and transfer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define lut for saggital to full mapping
        self.lut = lut
        # Options for parallel run
        if torch.cuda.device_count() > 0 and self.device == "cuda":
            self.model_parallel = True
        else:
            self.model_parallel = False

        # Initial model setup
        self.model = self.setup_model(cfg)
        self.model_name = self.cfg.MODEL.MODEL_NAME

        # Initial checkpoint loading
        self.load_checkpoint(ckpt)

    def setup_model(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg, train=False, device=self.device)  # ~ model = FastSurferCNN(params_network)
        model.to(self.device)

        if self.model_parallel:
            model = torch.nn.DataParallel(model)

        return model

    def set_cfg(self, cfg):
        self.cfg = cfg

    def set_model(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

        # Set up model
        model = build_model(self.cfg, device=self.device)
        model.to(self.device)

        if self.model_parallel:
            model = torch.nn.DataParallel(model)
        self.model = model

    def load_checkpoint(self, ckpt):
        logger.info("Loading checkpoint {}".format(ckpt))
        model_state = torch.load(ckpt, map_location=self.device)
        self.model.load_state_dict(model_state["model_state"], strict=False)

    def get_modelname(self):
        return self.model_name

    def get_cfg(self):
        return self.cfg

    def get_num_classes(self):
        return self.cfg.MODEL.NUM_CLASSES

    def get_plane(self):
        return self.cfg.DATA.PLANE

    def get_model_height(self):
        return self.cfg.MODEL.HEIGHT

    def get_model_width(self):
        return self.cfg.MODEL.WIDTH

    def get_max_size(self):
        if self.cfg.MODEL.OUT_TENSOR_WIDTH == self.cfg.MODEL.OUT_TENSOR_HEIGHT:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH
        else:
            return self.cfg.MODEL.OUT_TENSOR_WIDTH, self.cfg.MODEL.OUT_TENSOR_HEIGHT

    def get_device(self):
        return self.device

    @torch.no_grad()
    def eval(self, val_loader, pred_prob, out_scale=None):
        self.model.eval()

        # we should check here, whether the DataLoader is a Random or a SequentialSampler, but we cannot easily.
        if not isinstance(val_loader.sampler, torch.utils.data.SequentialSampler):
            logger.warning(
                "The Validation loader seems to not use the SequentialSampler. This might interfere with "
                "the assumed sorting of batches."
            )

        start_index = 0
        plane = self.cfg.DATA.PLANE
        # get output shape (except class dimension), for cutting prediction to correct size. Scenarios:
        # 1. Padding slices before running through the network --> Cut back to unpadded version (non-batch dimensions)
        # 2. Out_scale defined --> Leave batch dimension alone (will differ from other two), but cut others
        #                          in case we also have padding, pred_prob = output, output, output
        #                          To Do: How to correctly set shape for orig > scaled; leave out for now
        out_shapes = pred_prob.shape[:-1]

        from tqdm.contrib.logging import logging_redirect_tqdm
        from tqdm import tqdm
        with logging_redirect_tqdm():
            try:
                for batch_idx, batch in tqdm(enumerate(val_loader), total=len(val_loader), unit="batch"):

                    # Move data to the model device
                    images, scale_factors, affine = batch['image'].to(self.device), \
                                                    batch['scale_factor'].to(self.device), \
                                                    batch["affine"].to(self.device)

                    # Predict the current batch, output logits
                    pred = self.model(images, scale_factors,
                              affine=affine, inverse=affine,
                              scale_factor_out=out_scale)

                    if plane == "axial":
                        pred = pred.permute(3, 0, 2, 1)[0:out_shapes[0], :, 0:out_shapes[2], :]
                        pred_prob[:, start_index:start_index + pred.shape[1], :, :] += torch.mul(pred, 0.4)
                        start_index += pred.shape[1]

                    elif plane == "coronal":
                        pred = pred.permute(2, 3, 0, 1)[0:out_shapes[0], 0:out_shapes[1], ...]
                        pred_prob[:, :, start_index:start_index + pred.shape[2], :] += torch.mul(pred, 0.4)
                        start_index += pred.shape[2]

                    else:
                        pred = map_prediction_sagittal2full(pred, self.lut).permute(0, 3, 2, 1)[:, 0:out_shapes[1], 0:out_shapes[2], :]
                        pred_prob[start_index:start_index + pred.shape[0], :, :, :] += torch.mul(pred, 0.2)
                        start_index += pred.shape[0]
            except:
                logger.exception("Exception in batch {} of {} inference.".format(batch_idx, self.cfg.DATA.PLANE))
                raise
            else:
                logger.info("Inference on {} batches for {} successful".format(batch_idx + 1, self.cfg.DATA.PLANE))

        return pred_prob#, img

    def run(self, img_filename, orig_data, orig_zoom, out, out_res=None):
        # Set up DataLoader
        aug_trans = [ZeroPad((self.cfg.DATA.PADDED_SIZE, self.cfg.DATA.PADDED_SIZE),
                                              dim=self.cfg.DATA.DIMENSION),
                                      ToTensor(dim=self.cfg.DATA.DIMENSION)]

        transform = transforms.Compose(aug_trans)
        test_dataset = MultiScaleOrigDataThickSlices(img_filename, orig_data, orig_zoom, self.cfg,
                                                     transform=transform)

        test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                      batch_size=self.cfg.TEST.BATCH_SIZE)

        # Run evaluation
        start = time.time()
        out = self.eval(test_data_loader, out, out_scale=out_res)
        logger.info("Inference on {} finished in {:0.4f} seconds".format(img_filename, time.time()-start))

        return out

