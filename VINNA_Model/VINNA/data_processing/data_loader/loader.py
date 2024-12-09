# IMPORTS
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torchio as tio
import VINNA.data_processing.data_loader.dataset as dset
from VINNA.data_processing.augmentation.augmentation import ToTensor, ZeroPad, \
    AddGaussianNoise, RandomAffine, RandomLabelsToImage
import VINNA.utils.logging as logging_u

logger = logging_u.get_logger(__name__)


def get_dataloader(cfg, mode):
    """
        Creating the dataset and pytorch data loader
    :param cfg:
    :param mode: loading data for train, val and test mode
    :return:
    """
    assert mode in ['train', 'val'], f"dataloader mode is incorrect {mode}"

    padding_size = cfg.DATA.PADDED_SIZE
    dim = 3 if cfg.DATA.PLANE == "all" else 2

    if mode == 'train':

        if "None" in cfg.DATA.AUG:
            aug_trans = [ZeroPad((padding_size,) * dim, dim=dim), ToTensor(dim=dim)]

            if cfg.DATA.LATENT_AFFINE:
                # Already creates torch, so can be added after ToTensor
                aug_trans.append(RandomAffine([cfg.DATA.Rot_ANGLE_MIN, cfg.DATA.Rot_ANGLE_MAX],
                                              [cfg.DATA.TRANSLATION_MIN, cfg.DATA.TRANSLATION_MAX],
                                              ))

            # old transform
            if "Gaussian" in cfg.DATA.AUG:
                # Already creates torch.tensor, so can be added after ToTensor
                aug_trans.append(AddGaussianNoise(mean=0, std=0.1))

            transform = transforms.Compose(aug_trans)

            data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using standard Aug")

            dataset = dset.MultiScaleDatasetValIt(data_path, cfg, transform, mode=mode)
        else:
            logger.info(f"Torchio augmentations: {cfg.DATA.AUG}")
            # Crop around center mask and pad to final padding size (e.g. 256)
            crop_pad = tio.CropOrPad(target_shape=(7, 256, 256),
                                     padding_mode=0,
                                     mask_name="label",
                                     labels=[i for i in range(1, cfg.MODEL.NUM_CLASSES)],
                                     include=['img', 'label', 'weight', 't2_img'])
            # Elastic
            elastic = tio.RandomElasticDeformation(num_control_points=7,
                                                   max_displacement=(8, 8, 0),
                                                   locked_borders=2,
                                                   image_interpolation='linear',
                                                   include=['img', 'label', 'weight', 't2_img'])
            # Scales
            scaling = tio.RandomAffine(scales=(0.8, 1.15),
                                       degrees=0,
                                       translation=(0, 0, 0),
                                       isotropic=True,  # If True, scaling factor along all dimensions is the same
                                       center='image',
                                       default_pad_value='minimum',
                                       image_interpolation='linear',
                                       include=['img', 'label', 'weight', 't2_img'])

            # Rotation
            rot = tio.RandomAffine(scales=(1.0, 1.0),
                                   degrees=(180, 0, 0),
                                   translation=(0, 0, 0),
                                   isotropic=True,  # If True, scaling factor along all dimensions is the same
                                   center='image',
                                   default_pad_value='minimum',
                                   image_interpolation='linear',
                                   include=['img', 'label', 'weight', 't2_img'])

            # Translation
            tl = tio.RandomAffine(scales=(1.0, 1.0),
                                  degrees=0,
                                  translation=(0, 15.0, 15.0),
                                  isotropic=True,  # If True, scaling factor along all dimensions is the same
                                  center='image',
                                  default_pad_value='minimum',
                                  image_interpolation='linear',
                                  include=['img', 'label', 'weight', 't2_img'])

            # Random Anisotropy (Downsample image along an axis, then upsample back to initial space
            ra = tio.transforms.RandomAnisotropy(axes=(1, 2),
                                                 downsampling=(1.1, 1.5),
                                                 image_interpolation="linear",
                                                 include=["img", 't2_img', 'image_from_labels'])
            intensity_images = ["img", "t2_img"]
            if "SynthSeg" in cfg.DATA.AUG:
                # SynthSeg --> separate for T1 and T2?, set up mean and std for relevant labels
                synthseg = True
                intensity_images.append("image_from_labels")

                # Read in LUT information
                lut = pd.read_csv(cfg.DATA.SYNTHSEG_LUT, sep="\t")
                full_to_reduced = torch.tensor(lut["MappedID"].to_list())
                t2_std = lut["Std_" + cfg.DATA.IMG_TYPE].apply(
                    lambda x: tuple([float(x.split(",")[0][1:]), float(x.split(",")[1][:-1])])).to_list()
                t2_mean = lut["Mean_" + cfg.DATA.IMG_TYPE].apply(
                    lambda x: tuple([float(x.split(",")[0][1:]), float(x.split(",")[1][:-1])])).to_list()


                # Set up Augmentations (Mapping + Blurring to simulate PVEs)
                synthseg_transform = RandomLabelsToImage(label_key='label', image_key="image_from_labels",
                                                         used_labels=[i for i in range(1, cfg.MODEL.NUM_CLASSES)],
                                                         ignore_background=True, mean=t2_mean, std=t2_std,
                                                         mask_labels=full_to_reduced)
                blurring_transform = tio.transforms.RandomBlur(std=2.0, include=["image_from_labels"])
            else:
                synthseg = False

            # Bias Field
            bias_field = tio.transforms.RandomBiasField(coefficients=0.2, order=3, include=intensity_images)

            # Gamma
            random_gamma = tio.transforms.RandomGamma(log_gamma=(-0.1, 0.1), include=intensity_images)

            # Motion --> this is changing the image significantly, i.e. actually rotating it; artefacts do not appear
            motion = tio.transforms.RandomMotion(degrees=20, translation=20, num_transforms=20,
                                                 include=intensity_images)

            # Ghosting
            ghosting = tio.transforms.RandomGhosting(num_ghosts=(4, 10), axes=(1, 2), intensity=(0.5, 1),
                                                     restore=0.02, include=intensity_images)

            # Spiking
            spiking = tio.transforms.RandomSpike(num_spikes=1, intensity=1, include=intensity_images)

            # Blur
            blur = tio.transforms.RandomBlur(std=(0.0, 2.0), include=intensity_images)

            # Noise
            noise = tio.transforms.RandomNoise(mean=0, std=(0, 0.25), include=intensity_images)

            # Cut-out --> num iterations should not be too large (scattered image)
            cutout = tio.transforms.RandomSwap(patch_size=(15, 15, 1), num_iterations=10,
                                               include=intensity_images)

            all_augs = {"Elastic": [elastic, 0.8], "Scaling": [scaling, 0.8],
                        "Rotation": [rot, 0.8], "Translation": [tl, 0.8],
                        "RAnisotropy": [ra, 0.8], "BiasField": [bias_field, 0.6],
                        "RGamma": [random_gamma, 0.6], "CutOut": [cutout, 0.6],
                        "Motion": [motion, 0.6], "Ghosting": [ghosting, 0.6],
                        "Spike": [spiking, 0.6], "Noise": [noise, 0.6], "Blur": [blur, 0.6]}

            all_tfs_int = {all_augs[aug][0]: all_augs[aug][1] for aug in cfg.DATA.AUG if aug not in
                           ["SynthSeg", "Gaussian", "Elastic", "Scaling", "Rotation", "Translation"]}
            all_tfs_aff = {all_augs[aug][0]: all_augs[aug][1] for aug in cfg.DATA.AUG if
                           aug in ["Elastic", "Scaling", "Rotation", "Translation"]}

            gaussian_noise = True if "Gaussian" in cfg.DATA.AUG else False

            aug_order = [crop_pad, tio.Compose(all_tfs_aff, p=0.5)]
            if synthseg:
                aug_order.append(tio.Compose([synthseg_transform, blurring_transform]))
            aug_order.append(tio.Compose(all_tfs_int, p=0.5))

            transform = tio.Compose(aug_order,
                                    include=["img", "label", "weight", 't2_img', 'image_from_labels'])

            data_path = cfg.DATA.PATH_HDF5_TRAIN
            shuffle = True

            logger.info(f"Loading {mode.capitalize()} data ... from {data_path}. Using torchio Aug")

            dataset = dset.MultiScaleDatasetIt(data_path, cfg, gaussian_noise, transform)

    elif mode == 'val':
        data_path = cfg.DATA.PATH_HDF5_VAL
        shuffle = False
        aug_trans = [ZeroPad((padding_size,) * dim, dim=dim), ToTensor(dim=dim), ]
        transform = transforms.Compose(aug_trans)

        logger.info(f"Loading {mode.capitalize()} data ... from {data_path}")

        dataset = dset.MultiScaleDatasetValIt(data_path, cfg, transform, mode=mode)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        shuffle=shuffle,
        pin_memory=True,
    )

    return dataloader
