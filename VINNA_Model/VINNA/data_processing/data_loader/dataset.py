# IMPORTS
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio as tio
import time

import VINNA.data_processing.utils.data_utils as du
import VINNA.utils.logging as logging_u
import VINNA.data_processing.augmentation.augmentation as aug

logger = logging_u.getLogger(__name__)


# Operator to load imaged for inference
class MultiScaleOrigDataThickSlices(Dataset):
    """
    Class to load MRI-Image and process it to correct format for network inference
    """

    def __init__(self, img_filename, orig_data, orig_zoom, cfg, transform=None):
        assert transform is not None, "Transforms need to be defined for inference dataset (ZeroPad, ToTensor at least)"
        self.img_filename = img_filename
        self.plane = cfg.DATA.PLANE
        self.slice_thickness = cfg.MODEL.NUM_CHANNELS // 2
        self.base_res = cfg.MODEL.BASE_RES
        self.img_dimension = cfg.DATA.DIMENSION
        self.dim = cfg.DATA.DIMENSION
        self.affine_add = torch.tensor([0 for _ in range(self.dim)] + [1])
        if self.plane == "sagittal":
            orig_data = du.transform_sagittal(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Sagittal with input voxelsize {}".format(self.zoom))

        elif self.plane == "axial":
            orig_data = du.transform_axial(orig_data)
            self.zoom = orig_zoom[::-1][:2]
            logger.info("Loading Axial with input voxelsize {}".format(self.zoom))

        elif self.plane == "all":
            self.zoom = orig_zoom

        else:
            self.zoom = orig_zoom[:2]
            logger.info("Loading Coronal with input voxelsize {}".format(self.zoom))

        # Make 4D
        if self.plane != "all":
            # Create thick slices
            orig_thick = du.get_thick_slices(orig_data, self.slice_thickness)
            assert orig_thick.max() > 0.8, f"Multi Dataset - orig thick fail, max removed {orig_thick.max()}"
            orig_thick = np.transpose(orig_thick, (2, 0, 1, 3))  # D=B x H x W x C
            assert orig_thick.max() > 0.8, "Multi Dataset - transpose fail, max removed  maximum {} minimum {}".format(
                orig_thick.max(), orig_thick.min())
        else:
            orig_thick = np.expand_dims(orig_data, axis=[0, -1])  # create channel dimension (B=1 x D x H x W x C=1)
        self.images = orig_thick
        self.count = self.images.shape[0]
        self.transforms = transform
        self.latent_aug = cfg.DATA.LATENT_AFFINE

        logger.info(f"Successfully loaded Image from {img_filename}")

    def _get_scale_factor(self):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        scale = self.base_res / np.asarray(self.zoom)
        return scale

    def get_default_affine(self):
        # exclude sf, because scaling does not work correctly with it in affine in stn models
        if self.plane != "all":
            return np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            return np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    def __getitem__(self, index):
        img = self.images[index]

        scale_factor = self._get_scale_factor()
        affine = self.get_default_affine()
        # inverse of identity matrix is identiy matrix, affine is not applied during inference
        sample = self.transforms({'img': img, 'scale_factor': scale_factor,
                                  'affine': affine, 'inverse': affine})

        return {'image': sample['img'], 'scale_factor': sample['scale_factor'],
                'affine': sample['affine'], 'inverse': sample['inverse']}

    def __len__(self):
        return self.count


# Operator to load hdf5-file for training
class MultiScaleDatasetIt(Dataset):
    """
    Class for loading aseg file with augmentations (transforms), images are only loaded into memory upon execution
    (= no pre-loading)

    Used if:
     1. Training if augmentations defined in cfg.DATA.AUG (i.e. any external augmentations, RandomElasticDeformation,
     RandomAffine (scaling, rotation, Translation), RansomAnisotropy, RandomBiasField, RandomGamma, RandomSwap =cutout)
    """

    def __init__(self, dataset_path, cfg, gn_noise=False, transforms=None):

        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.gn_noise = gn_noise
        self.load_to_memory = False
        self.sizes = cfg.DATA.SIZES
        self.latent_aug = cfg.DATA.LATENT_AFFINE
        self.upr_affine = cfg.DATA.UPR_AFFINE
        self.elastic = cfg.MODEL.ELASTIC_DEFORMATION
        self.cfg = cfg
        self.plane = cfg.DATA.PLANE
        self.tl_min, self.tl_max = cfg.DATA.TRANSLATION_MIN, cfg.DATA.TRANSLATION_MAX
        self.rot_min, self.rot_max = cfg.DATA.Rot_ANGLE_MIN, cfg.DATA.Rot_ANGLE_MAX
        self.dim = 3 if cfg.DATA.PLANE == "all" else 2
        self.affine_add = torch.tensor([0 for _ in range(self.dim)] + [1])
        # Load the h5 file and save it to the datasets
        self._data = {"images": [], "labels": [], "weights": [], "subjects": [],
                      "zooms": [], "t2_images": [], "upr_affine": []}
        self._params = {"img_name": "orig_dataset", "t2_name": "t2_dataset",
                        "labels": "aseg_dataset", "weights": "weight_dataset",
                        "zoom": "zoom_dataset", "upr_aff_name": "upr_affine_dataset",
                        "subject_name": "subject"}

        # Open file in reading mode
        start = time.time()
        try:
            # force conversion to iterable
            try:
                file_iter = iter([dataset_path] if isinstance(dataset_path, str) else dataset_path)
            except TypeError as e:
                raise TypeError("dataset_path has to be a string or an iterable list of strings")

            # open file in reading mode
            for dset_file in file_iter:
                hf = h5py.File(dset_file, "r")
                # with h5py.File(dset_file, "r") as hf: # can not be used because it would close the file
                self._load(hf)

            self.sizes = self._h5length(self._data["images"])
            self.count = np.sum(self.sizes)
            self.cumsizes = np.cumsum(self.sizes)
            # self.count = np.sum(self._h5length(self._data["images"]))
            print(self.count)
            # self.count = len(self.images)
            self.transforms = transforms

            logger.info("Successfully loaded {self.count} data from {dataset_path} "
                        "with plane {cfg.DATA.PLANE} in {time.time()-start} seconds")
            # except Exception as e:
            #     print("Loading failed: {}".format(e))

        except NotImplementedError as e:
            print(e)

    @staticmethod
    def _h5length(h5_list):
        """Static function to get number of samples from list."""
        return [h5_list[i].shape[0] for i in range(len(h5_list))]

    @staticmethod
    def _init_field(num: int, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            return np.empty_like((num,) + src.shape[1:])
        else:
            return [""] * num

    def _load(self, hf):
        """Iteratively loads files into memory."""

        source_to_target_img = {
            self._params["img_name"]: "images",
            self._params["subject_name"]: "subjects",
            self._params["labels"]: "labels",
            self._params["weights"]: "weights",
            self._params["zoom"]: "zooms",
            self._params["t2_name"]: "t2_images",
            self._params["upr_aff_name"]: "upr_affine"
        }
        self._source_to_target_assign(source_to_target_img, hf)

    def _source_to_target_assign(self, source_to_target, root):
        """Helper function to copy hdf5 data to ram, if required."""
        for s, t in source_to_target.items():
            for size in self.sizes[::-1]:
                try:
                    data = root[f'{size}'][s]
                except KeyError as e:
                    print(f"KeyError: {e}. "
                          f"=object {size} of type {type(size)} does not exist or "
                          f"dataset part {s} is not existent in the hdf5-keys")
                    continue

                if self.load_to_memory:
                    data = np.asarray(data)
                self._data[t].append(data)

    def get_subject_names(self):
        return self.subjects

    def _get_scale_factor(self, img_zoom, scale_aug):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        # Add scaling parameter from external scaling augmentation, if any
        if torch.all(scale_aug > 0):
            img_zoom *= (1 / scale_aug)

        if torch.any(img_zoom <= 0.1) or torch.any(img_zoom >= 10):
            img_zoom = torch.tensor([self.base_res for _ in range(len(img_zoom))], dtype=img_zoom.dtype)

        scale = self.base_res / img_zoom

        if self.gn_noise:
            scale += torch.randn(1) * 0.1 + 0

        return scale

    def _pad(self, image):
        if len(image.shape) == self.dim:
            padded_img = np.zeros((self.max_size,) * self.dim, dtype=image.dtype)
        else:
            padded_img = np.zeros((self.max_size,) * self.dim + (image.shape[-1],), dtype=image.dtype)

        if self.dim == 2:
            padded_img[0: image.shape[0], 0: image.shape[1]] = image
        else:
            padded_img[0: image.shape[0], 0: image.shape[1], 0: image.shape[2]] = image
        return padded_img

    def unify_imgs(self, index_slice):  # img, label, weight):

        img = self._pad(self._get_sample(index_slice, "images"))
        t2 = self._pad(self._get_sample(index_slice, "t2_images"))
        label = self._pad(self._get_sample(index_slice, "labels"))
        weight = self._pad(self._get_sample(index_slice, "weights"))

        return img, label, weight, t2

    def get_all_images(self, index_slice):
        img = self._get_sample(index_slice, "images")
        t2 = self._get_sample(index_slice, "t2_images")
        label = self._get_sample(index_slice, "labels")
        weight = self._get_sample(index_slice, "weights")

        return img, label, weight, t2

    def get_affine(self, index_slice):
        return self._get_sample(index_slice, "upr_affine")

    def _get_sample(self, index_slice, key):
        return self._index(self._data[key], index_slice)

    def _index(self, h5_list, index):
        if isinstance(index, int):
            return self._index_list(h5_list, [index])
        elif isinstance(index, list):
            return self._index_list(h5_list, index)
        else:
            raise ValueError("index should be either int or list, but was %s." % type(index).__name__)

    def _index_list(self, h5_list, index):
        # create empty array with dimensions batch, h,w,c
        out = np.empty(h5_list[0].shape[1:])

        # list with dimension of images (256)
        # determine which list to query: #208050+6228+7430
        j = 0 if index < self.cumsizes[0] else 1 if index < self.cumsizes[1] else 2

        index[0] -= self.cumsizes[j - 1] if j != 0 else 0
        out = self._assign_field(0, index, out, h5_list[j])
        return out

    def _assign_field(self, t_index, s_index, out, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            if self._params["load_to_memory"]:
                out[t_index] = src[s_index]
            else:
                out[t_index] = np.asarray(src[s_index]).squeeze()
        else:
            out = src[s_index].squeeze()
        return out

    def get_default_affine(self):
        if self.plane != "all":
            a = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
        else:
            a = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)
        return a, a

    def get_random_affine(self):
        tl = aug.get_random_translation([self.tl_min, self.tl_max], self.dim)
        rot = aug.get_random_rotation([self.rot_min, self.rot_max], self.dim)

        rand_affine = tl @ rot
        rand_inverse = torch.inverse(rand_affine)

        return rand_affine[:self.dim, :self.dim + 1], rand_inverse[:self.dim, :self.dim + 1]

    def get_upright_affine(self, affine_info, vol_shape):
        a = torch.as_tensor(du.affine_pytorch_style(affine_info, self.plane,
                                                    vol_shape, vol_shape, normalized=False), dtype=torch.float32)
        a_inv = torch.inverse(torch.vstack((a, self.affine_add)))
        return a, a_inv[..., :self.dim, :self.dim + 1]

    def __getitem__(self, index):
        # padded_img, padded_label, padded_weight = self.unify_imgs(index) # , padded_t2
        padded_img, padded_label, padded_weight, padded_t2 = self.get_all_images(index)
        if self.plane != "all":
            img = np.expand_dims(padded_img.transpose((2, 0, 1)), axis=0)  # C=7, H, W, D=1// 1, 7, 256, 256 if axis=0

            t2_img = np.expand_dims(padded_t2.transpose((2, 0, 1)), axis=0)  # C, H, W, D=1
            # Get thick slices for the labels (simple work around for synthseg right now)
            label = np.tile(padded_label[np.newaxis, :, :], (1, 7, 1, 1))  # C=1, H, W, D=1 //1, 7, 256, 256 if tile
            weight = np.tile(padded_weight[np.newaxis, :, :], (1, 7, 1, 1))
            zoom_aug = torch.as_tensor([0., 0.])
        else:
            img = padded_img.transpose((3, 0, 1, 2))  # img = C = 1, H, W, D: channel first in torch, last in numpy
            t2_img = padded_t2.transpose((3, 0, 1, 2))
            label = padded_label[np.newaxis, ...]
            weight = padded_weight[np.newaxis, ...]
            zoom_aug = torch.as_tensor([0., 0., 0.])
        assert img.shape == (1, 7, 256, 256), f"wrong shape {img.shape}"
        assert label.shape == (1, 7, 256, 256), f"wrong shape {label.shape}"
        assert not np.any(np.isnan(img))
        assert not np.any(np.isnan(t2_img))
        assert not np.any(np.isnan(label))
        subject = tio.Subject({'img': tio.ScalarImage(tensor=img),
                               't2_img': tio.ScalarImage(tensor=t2_img),
                               'label': tio.LabelMap(tensor=label),
                               'weight': tio.LabelMap(tensor=weight)}
                              )

        if self.transforms is not None:
            tx_sample = self.transforms(subject)  # this returns data as torch.tensors
            img = tx_sample['img'].data[0, ...].float()  # .astype(np.float32)
            t2_img = tx_sample['t2_img'].data[0, ...].float()  # .astype(np.float32)
            label = tx_sample['label'].data[0, 3, ...].byte()  # .astype(np.uint8)
            weight = tx_sample['weight'].data[0, 3, ...].float()  # .astype(np.float32)

            assert label.shape == torch.Size([256, 256]), f"wrong shape {label.shape}"

            if "image_from_labels" in tx_sample.keys() and torch.rand(1) > self.cfg.DATA.SYNTHSEG_TRADEOFF:
                if self.cfg.DATA.IMG_TYPE == "image":
                    img = tx_sample["image_from_labels"].data[0, ...].float()
                else:
                    t2_img = tx_sample["image_from_labels"].data[0, ...].float()
            assert t2_img.shape == torch.Size([7, 256, 256]), f"wrong shape {t2_img.shape}"

            # get updated scalefactor, in case of scaling, not ideal - fails if scales is not in dict
            rep_tf = tx_sample.get_composed_history()
            if rep_tf:
                try:
                    if self.plane != "all":
                        zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])[:-1]
                    else:
                        zoom_aug += torch.as_tensor(rep_tf[0]._get_reproducing_arguments()["scales"])
                except KeyError:
                    pass
            # Normalize image and clamp between 0 and 1
            img = torch.nan_to_num(torch.clamp(img / img.max(), min=0.0, max=1.0))
            t2_img = torch.nan_to_num(torch.clamp(t2_img / t2_img.max(), min=0.0, max=1.0))
            assert not torch.any(torch.isnan(img))
            assert not torch.any(torch.isnan(t2_img))
            assert not torch.any(torch.isnan(label))
        scale_factor = self._get_scale_factor(torch.from_numpy(self._get_sample(index, "zooms")), scale_aug=zoom_aug)

        if self.latent_aug and torch.rand(1) < 0.5:
            affine, inverse = self.get_random_affine()
        elif self.upr_affine:
            affine, inverse = self.get_upright_affine(self.get_affine(index), img.shape[-2:])
        else:
            affine, inverse = self.get_default_affine()

        return {'image': img, 'label': label, 'weight': weight, 't2_image': t2_img,
                "scale_factor": scale_factor, 'affine': affine, 'inverse': inverse}

    def __len__(self):
        return self.count


# Operator to load hdf5-file for validation
class MultiScaleDatasetValIt(Dataset):
    """
    Class for loading aseg file with augmentations (transforms), images are only loaded into memory upon execution
    (= no pre-loading)

    Used if:
     1. Training if None in cfg.DATA.AUG (i.e. if latent_affine or Gaussian_Noise will be added,
        without external augmentations)
     2. Validation (always as no augmentation is applied)
    """

    def __init__(self, dataset_path, cfg, transforms=None, mode="val"):

        self.max_size = cfg.DATA.PADDED_SIZE
        self.base_res = cfg.MODEL.BASE_RES
        self.fix_res = 2.0
        self.cfg = cfg
        self.latent_aug = cfg.DATA.LATENT_AFFINE if mode == "train" else False  # always set to valse in validation
        self.upr_affine = cfg.DATA.UPR_AFFINE
        self.load_to_memory = False
        self.sizes = cfg.DATA.SIZES
        self.plane = cfg.DATA.PLANE
        self.dim = 2 if cfg.DATA.PLANE != "all" else 3
        self.affine_add = torch.tensor([0 for _ in range(self.dim)] + [1])
        self.elastic = cfg.MODEL.ELASTIC_DEFORMATION if mode == "train" else False
        # Load the h5 file and save it to the dataset
        self._data = {"images": [], "labels": [], "weights": [], "subjects": [],
                      "zooms": [], "t2_images": [], "upr_affine": []}
        self._params = {"img_name": "orig_dataset", "labels": "aseg_dataset", "weights": "weight_dataset",
                        "zoom": "zoom_dataset", "subject_name": "subject", "t2_name": "t2_dataset",
                        "upr_aff_name": "upr_affine_dataset"}

        # Open file in reading mode
        start = time.time()
        try:
            # force conversion to iterable
            try:
                file_iter = iter([dataset_path] if isinstance(dataset_path, str) else dataset_path)
            except TypeError as e:
                raise TypeError("dataset_path has to be a string or an iterable list of strings")
            # file_iter = dataset_path
            # open file in reading mode
            for dset_file in file_iter:
                hf = h5py.File(dset_file, "r")
                self._load(hf)

            self.sizes = self._h5length(self._data["images"])
            self.count = np.sum(self.sizes)
            self.cumsizes = np.cumsum(self.sizes)

            print(self.count)
            self.transforms = transforms

            logger.info(f"Successfully loaded {self.count} data from {dataset_path} "
                        "with plane {cfg.DATA.PLANE} in {time.time()-start} seconds")

        # except Exception as e:
        #     print("Loading failed: {}".format(e))
        except NotImplementedError as e:
            print(e)

    @staticmethod
    def _h5length(h5_list):
        """Static function to get number of samples from list."""
        return [h5_list[i].shape[0] for i in range(len(h5_list))]

    @staticmethod
    def _init_field(num: int, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            return np.empty_like((num,) + src.shape[1:])
        else:
            return [""] * num

    def _load(self, hf):
        """Iteratively loads files into memory."""

        source_to_target_img = {
            self._params["img_name"]: "images",
            self._params["subject_name"]: "subjects",
            self._params["labels"]: "labels",
            self._params["weights"]: "weights",
            self._params["zoom"]: "zooms",
            self._params["t2_name"]: "t2_images",
            self._params["upr_aff_name"]: "upr_affine"
        }
        self._source_to_target_assign(source_to_target_img, hf)

    def _source_to_target_assign(self, source_to_target, root):
        """Helper function to copy hdf5 data to ram, if required."""
        for s, t in source_to_target.items():
            for size in self.sizes[::-1]:
                try:
                    data = root[f'{size}'][s]
                except KeyError as e:
                    print(f"KeyError: {e}. "
                          f"=object {size} of type {type(size)} does not exist or "
                          f"dataset part {s} is not existent in the hdf5-keys")
                    continue

                if self.load_to_memory:
                    data = np.asarray(data)
                self._data[t].append(data)
                # self._data[t].extend(data) --> loads data into memory

    def get_subject_names(self):
        return self.subjects

    def get_affine(self, index_slice):
        return self._get_sample(index_slice, "upr_affine")

    def _get_scale_factor(self, img_zoom):
        """
        Get scaling factor to match original resolution of input image to
        final resolution of FastSurfer base network. Input resolution is
        taken from voxel size in image header.

        ToDO: This needs to be updated based on the plane we are looking at in case we
        are dealing with non-isotropic images as inputs.
        :param img_zoom:
        :return np.ndarray(float32): scale factor along x and y dimension
        """
        # Zoom is sometimes incorrectly set to a gigantic, negative number, or zero!
        # Potentially issues with the file header
        # ToDo: Fix this in the generate_data_hdf5.py script
        if np.any(img_zoom <= 0.1) or np.any(img_zoom >= 10):
            img_zoom = np.asarray([self.base_res for _ in range(len(img_zoom))], dtype=img_zoom.dtype)
        scale = self.base_res / img_zoom

        return scale

    def _get_sample(self, index_slice, key):
        return self._index(self._data[key], index_slice)

    def _index(self, h5_list, index):
        if isinstance(index, int):
            return self._index_list(h5_list, [index])
        elif isinstance(index, list):
            return self._index_list(h5_list, index)
        else:
            raise ValueError("index should be either int or list, but was %s." % type(index).__name__)

    def _index_list(self, h5_list, index):
        # create empty array with dimensions batch, h,w,c
        out = np.empty(h5_list[0].shape[1:])

        # list with dimension of images (256)
        # determine which list to query: #208050+6228+7430
        j = 0 if index < self.cumsizes[0] else 1 if index < self.cumsizes[1] else 2

        index[0] -= self.cumsizes[j - 1] if j != 0 else 0

        out = self._assign_field(0, index, out, h5_list[j])
        return out

    def _assign_field(self, t_index, s_index, out, src):
        if h5py.check_dtype(vlen=src.dtype) is not None:
            if self._params["load_to_memory"]:
                out[t_index] = src[s_index]
            else:
                out[t_index] = np.asarray(src[s_index]).squeeze()
        else:
            out = src[s_index].squeeze()
        return out

    def get_default_affine(self):
        if self.dim == 2:
            return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        else:
            return np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def get_upright_affine(self, affine_info, vol_shape):
        a = torch.as_tensor(du.affine_pytorch_style(affine_info, self.plane,
                                                    vol_shape, vol_shape, normalized=False), dtype=torch.float32)
        a_inv = torch.inverse(torch.vstack((a, self.affine_add)))
        return a, a_inv[..., :self.dim, :self.dim + 1]

    def __getitem__(self, index):
        img = self._get_sample(index, "images")  # self.images[index]
        label = self._get_sample(index, "labels")  # self.labels[index]
        weight = self._get_sample(index, "weights")  # self.weights[index]
        try:
            t2 = self._get_sample(index, "t2_images")
        except IndexError as e:
            t2 = img
        scale_factor = self._get_scale_factor(
            self._get_sample(index, "zooms"))  # data["zooms"])#self._get_sample(index, "zooms"))
        affine = self.get_default_affine()
        inverse = affine  # inverse of identity is identity

        if self.transforms is not None:
            tx_sample = self.transforms({'img': img, 'label': label, 'weight': weight, 'inverse': inverse,
                                         'scale_factor': scale_factor, 't2': t2, 'affine': affine})

            img = tx_sample['img']
            label = tx_sample['label']
            weight = tx_sample['weight']
            scale_factor = tx_sample['scale_factor']
            t2 = tx_sample['t2']

            # Apply random rotation with probability 0.5
            if self.latent_aug and torch.rand(1) < 0.5:
                affine, inverse = tx_sample["affine"], tx_sample["inverse"]
            elif self.upr_affine:
                affine, inverse = self.get_upright_affine(self.get_affine(index), img.shape[-2:])
            else:
                affine, inverse = torch.from_numpy(affine), torch.from_numpy(inverse)
        return {'image': img, 'label': label, 'weight': weight, 'inverse': inverse,
                'scale_factor': scale_factor, 'affine': affine, 't2_image': t2}

    def __len__(self):
        return self.count



