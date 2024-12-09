# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import numpy as np
import math
import torch
import torchio as tio
import torchvision.transforms as tt
import typing as _T

##
# Static helper functions
##

# Rotation
def get_random_rotation(degrees: _T.Sequence, dim: int) -> torch.Tensor:
    ea0 = random_uniform_torch(degrees[0], degrees[1])
    return torch.tensor([[math.cos(ea0), -math.sin(ea0), 0],
                         [math.sin(ea0), math.cos(ea0), 0],
                         [0, 0, 1]])


def rotation_around_point(rotation: torch.Tensor, point: _T.Union[_T.Sequence, float]) -> torch.Tensor:
    if isinstance(point, list):
        point = np.asarray(point)
    # get forward point translation
    tl_fw = get_translation_to_point(-point)
    tl_back = get_translation_to_point(point)

    # Combine translation and rotation to affine matrix
    return tl_fw @ rotation @ tl_back


def get_translation_to_point(point: _T.Union[_T.Sequence, float]) -> torch.Tensor:
    if isinstance(point, list):
        point = np.asarray(point)

    dim = point.shape[-1]
    # Create affine identity matrix (one dim more than point coords)
    tl = torch.eye(n=dim + 1)
    # Add affinity 1 to end of point coordinate and overwrite identiy matrix with it
    tl[0:dim, dim] = torch.tensor(point)
    return tl


# Translation
def get_random_translation(tlrange: _T.Sequence, dim: int) -> torch.Tensor:
    tr_x = random_uniform_torch(tlrange[0], tlrange[1])
    tr_y = random_uniform_torch(tlrange[0], tlrange[1])
    return torch.tensor([[1, 0, -tr_x],
                         [0, 1, -tr_y],
                         [0, 0, 1]])


# Random number from uniform distribution with torch
def random_uniform_torch(r1, r2):
    return (r1 - r2) * torch.rand(1) + r2


##
# Transformations for training
##
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __init__(self, dim=2):
        self.dim = dim

    def norm_and_transpose(self, img):

        if img.dtype == np.dtype('u1'):
            img = img.astype(np.float32)
            # Normalize image and clamp between 0 and 1
            img = img.astype(np.float32)
            img = np.clip(img / 255.0, a_min=0.0, a_max=1.0)

        # swap color axis because
        # numpy image: H x W x C, D x H x W x C (=channel last)
        # torch image: C X H X W , C x D x H x W (=channel first)
        if self.dim == 2:
            img = img.transpose((2, 0, 1))
        elif self.dim == 3 and len(img.shape) == 3:
            # D x H x W -> C x D x H x W
            img = np.expand_dims(img, axis=0)
        else:
            # D x H x W x C -> C x D x H x W
            img = img.transpose((3, 0, 1, 2))
        return img

    def __call__(self, sample):

        for name, img in sample.items():
            if name not in ['label', 'weight', 'scale_factor', 'affine', 'inverse', 'ctrlp', 'inv_ctrlp']:
                img = self.norm_and_transpose(img)
            sample[name] = torch.from_numpy(img)

        return sample


# Extension of RandomLabelsToImage
# 1. Change indexing (fails if not all classes are present in the slice)
# 2. Map cortex and White matter subparcels to one class
class RandomLabelsToImage(tio.RandomLabelsToImage):

    def __init__(
            self,
            label_key: _T.Optional[str] = None,
            used_labels: _T.Optional[_T.Sequence[int]] = None,
            image_key: str = 'image_from_labels',
            mean: _T.Optional[_T.Sequence[_T.Tuple]] = None,
            std: _T.Optional[_T.Sequence[_T.Tuple]] = None,
            default_mean: _T.Tuple = (0.1, 0.9),
            default_std: _T.Tuple = (0.01, 0.1),
            discretize: bool = False,
            ignore_background: bool = False,
            mask_labels: _T.Optional[torch.Tensor] = None,
            **kwargs,
    ):
        super().__init__(label_key, used_labels, image_key, mean, std, default_mean, default_std,
                         discretize, ignore_background, **kwargs)
        self.mask_labels = mask_labels

    def parse_gaussian_parameters(
        self,
        params: _T.Sequence[_T.Tuple],
        name: str,
    ) -> _T.List[_T.Tuple]:
        tio.utils.check_sequence(params, name)
        params = [
            self.parse_gaussian_parameter(p, f'{name}[{i}]')
            for i, p in enumerate(params)
        ]
        if self.used_labels is not None:
            message = (
                f'If both "{name}" and "used_labels" are defined, '
                f'they must have the same length or the same max class. '
                f'Got Params length {len(params)}, Labels lenght {len(self.used_labels)}'
                f'Max labels {max(self.used_labels)}'
            )
            assert len(params) == len(self.used_labels) or len(params) > max(self.used_labels), message
        return params


    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Overwrite apply transforms to avoid wrong label assignments
        :param tio.Subject subject: Torchio Subject (with images, labels, weights)
        :return tio.Subject: transformed Subject (image from labels)
        """
        if self.label_key is None:
            iterable = subject.get_images_dict(intensity_only=False).items()
            for name, image in iterable:
                if isinstance(image, tio.LabelMap):
                    self.label_key = name
                    break
            else:
                message = f'No label maps found in subject: {subject}'
                raise RuntimeError(message)

        arguments = {
            'label_key': self.label_key,
            'mean': [],
            'std': [],
            'image_key': self.image_key,
            'used_labels': self.used_labels,
            'discretize': self.discretize,
            'ignore_background': self.ignore_background,
        }

        label_map = subject[self.label_key].data
        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        if self.mask_labels is not None:
            if not is_discretized:
                raise RuntimeError(f"Error, data is not discretize, so reduced mapping can not be used "
                                   f"(only implemented for discrete label maps)")
            assert torch.max(label_map.ravel().int()) < len(self.mask_labels), f"Does not match: {torch.max(label_map.ravel().int())}, {self.mask_labels.shape}"
            label_map = torch.reshape(torch.index_select(self.mask_labels, 0, label_map.ravel().int()), label_map.shape)
            reduced_labels = tio.LabelMap(affine=subject[self.label_key].affine, tensor=label_map)
            subject.add_image(reduced_labels, "reduced_labels")
            arguments["label_key"] = "reduced_labels"

        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        # --> checks length only in original implementation, some labels might however not be present
        # in all batches; better to check if largest label equals or is smaller then length of mean and std
        _check_mean_and_std_length(labels, self.mean, self.std)  # type: ignore[arg-type]  # noqa: B950

        for label in labels:
            # Index with label - 1 (if we do not have info for background class 0)
            assert len(self.mean) > label, f"Mean does not fit label: {len(self.mean)}, {label}"
            mean, std = self.get_params(label)
            means = arguments['mean']
            stds = arguments['std']
            assert isinstance(means, list)
            assert isinstance(stds, list)
            # Append mean and std for present labels only, hence in LabelsToImage, the call
            # is correct whether all labels are present.
            means.append(mean)
            stds.append(std)

        transform = tio.LabelsToImage(**self.add_include_exclude(arguments))
        transformed = transform(subject)
        assert isinstance(transformed, tio.Subject)
        return transformed


def _check_mean_and_std_length(
    labels: _T.Sequence[int],
    means: _T.Optional[_T.Sequence[_T.Tuple]],
    stds: _T.Optional[_T.Sequence[_T.Tuple]],
) -> None:
    """
    Checks length only in original implementation, some labels might however not be present
    in all batches; better to check if largest label equals or is smaller then length of mean and std
    :param _T.Sequence(int) labels: list of labels in batch (np.unique from labelMap)
    :param _T.Sequence(_T.Tuple) means: list of mean range (tuple(start, stop)) for each label
    :param _T.Sequence(_T.Tuple) stds: list of std range (tuple(start, stop)) for each label
    :return: None, raises error if length of means/std is unequal to length of labels
    """
    num_labels = len(labels)
    max_labels = max(labels)
    if means is not None:
        num_means = len(means)
        message = (
            '"mean" must define a value for each label but length of "mean"'
            f' is {num_means} while {num_labels} labels with maximum {max_labels} were found.'
            f'All labels: {labels}'
        )
        # add additional check that num_means is not smaller than max_labels for cases
        # where consecutive labels are used and not all are in every slice
        if num_means < max_labels:
            raise RuntimeError(message)
    if stds is not None:
        num_stds = len(stds)
        message = (
            '"std" must define a value for each label but length of "std"'
            f' is {num_stds} while {num_labels} labels were found'
        )
        if num_stds != num_labels and num_stds < max_labels:
            raise RuntimeError(message)


class ZeroPad2DTorchio(tio.SpatialTransform):
    def __init__(self, output_size, pos="top_left", **kwargs):
        super().__init__(**kwargs)
        if isinstance(output_size, float):
            output_size = (output_size,) * 2
        self.output_size = output_size
        self.pos = pos

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            data = image.numpy()
            c, h, w, d = data.shape
            print("Image data", data.shape, data.dtype)
            padded_img = np.zeros((c, self.output_size, self.output_size, d), dtype=data.dtype)

            if self.pos == "top_left":
                padded_img[:, 0:h, 0:w, :] = data

            else:
                raise ValueError("Only top_left padding supported. Other padding positions are not implemented.")

            image.set_data(torch.from_numpy(padded_img))

        return subject


class ZeroPad(object):
    def __init__(self, output_size, pos='top_left', dim=2):
        """
         Pad the input with zeros to get output size
        :param output_size:
        :param pos: position to put the input
        """
        if isinstance(output_size, float):
            output_size = (output_size,) * dim
        self.output_size = output_size
        self.pos = pos
        self.dim = dim

    def _pad(self, image):
        if len(image.shape) == self.dim:
            padded_img = np.zeros(self.output_size, dtype=image.dtype)
        elif len(image.shape) > self.dim:
            padded_img = np.zeros(self.output_size + (image.shape[-1],), dtype=image.dtype)

        if self.pos == 'top_left':
            if self.dim == 2:
                padded_img[0: image.shape[0], 0: image.shape[1]] = image
            else:
                padded_img[0: image.shape[0], 0: image.shape[1], 0: image.shape[2]] = image
        return padded_img

    def __call__(self, sample):

        for name, img in sample.items():
            if name not in ['scale_factor', 'affine', 'inverse', 'ctrlp', 'inv_ctrlp']:
                sample[name] = self._pad(img)

        return sample


class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        # change 1 to sf.size() for isotropic scale factors (now same noise change added to both dims)
        sample["scale_factor"] = sample['scale_factor'] + torch.randn(1) * self.std + self.mean
        return sample


class RandomAffine:

    def __init__(self, rotation, translation):
        self.zero = torch.tensor([[0, 0, 1]], dtype=torch.float32)
        self.tlmin = translation[0]
        self.tlmax = translation[1]
        self.rotmin = rotation[0]
        self.rotmax = rotation[1]

    # Rotation
    @staticmethod
    def get_rotation(ea0):
        return torch.tensor([[math.cos(ea0), -math.sin(ea0), 0],
                             [math.sin(ea0), math.cos(ea0), 0],
                             [0, 0, 1]])

    # Translation
    @staticmethod
    def get_translation(tr_x, tr_y):
        return torch.tensor([[1, 0, -tr_x],
                             [0, 1, -tr_y],
                             [0, 0, 1]])

    @staticmethod
    def random_uniform_torch(r1, r2):
        return (r1 - r2) * torch.rand(1) + r2

    # get combined transform
    def get_random_affine(self, degrees=[0, 90], tlrange=[0, 1], channels=1):
        # scale = self.get_scaling(sf)
        tl = torch.stack([self.get_translation(self.random_uniform_torch(tlrange[0], tlrange[1]),
                                               self.random_uniform_torch(tlrange[0], tlrange[1])) for _ in
                          range(channels)],
                         dim=0)
        rot = torch.stack(
            [self.get_rotation(self.random_uniform_torch(degrees[0], degrees[1])) for _ in range(channels)],
            dim=0)
        affine = tl @ rot
        inverse = torch.inverse(affine)
        return affine[0, 0:2], inverse[0, 0:2]  # .to(self.device)

    # reverse transform
    def prepare_inverse(self, affine_test):
        n = affine_test.size()[0]

        # Concat to get affine shape 3x3 for each batch
        affine_ = torch.stack(tuple(torch.cat((affine_test[i, ...], self.zero), dim=0)
                                    for i in range(n)))
        inverse_ = torch.inverse(affine_)

        # Remove affine row to get each batch to dim 2x3
        fin = torch.stack(tuple(inverse_[i, 0:2, 0:3] for i in range(n)))
        return fin

    def __call__(self, sample):
        sample["affine"], sample["inverse"] = self.get_random_affine(degrees=[self.rotmin, self.rotmax],
                                                                     tlrange=[self.tlmin, self.tlmax],
                                                                     channels=1)
        return sample


class AugmentationPadImage(object):
    """
    Pad Image with either zero padding or reflection padding of img, label and weight
    """

    def __init__(self, pad_size=((16, 16), (16, 16)), pad_type="edge"):

        assert isinstance(pad_size, (int, tuple))

        if isinstance(pad_size, int):

            # Do not pad along the channel dimension
            self.pad_size_image = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
            self.pad_size_mask = ((pad_size, pad_size), (pad_size, pad_size))

        else:
            self.pad_size = pad_size

        self.pad_type = pad_type

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']

        sample["img"] = np.pad(img, self.pad_size_image, self.pad_type)
        sample["label"] = np.pad(label, self.pad_size_mask, self.pad_type)
        sample["weight"] = np.pad(weight, self.pad_size_mask, self.pad_type)

        return sample


class AugmentationRandomCrop(object):
    """
    Randomly Crop Image to given size
    """

    def __init__(self, output_size, crop_type='Random'):

        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            self.output_size = output_size

        self.crop_type = crop_type

    def __call__(self, sample):
        img, label, weight = sample['img'], sample['label'], sample['weight']

        h, w, _ = img.shape

        if self.crop_type == 'Center':
            top = (h - self.output_size[0]) // 2
            left = (w - self.output_size[1]) // 2

        else:
            top = np.random.randint(0, h - self.output_size[0])
            left = np.random.randint(0, w - self.output_size[1])

        bottom = top + self.output_size[0]
        right = left + self.output_size[1]

        # print(img.shape)
        sample["img"] = img[top:bottom, left:right, :]
        sample["label"] = label[top:bottom, left:right]
        sample["weight"] = weight[top:bottom, left:right]

        return sample


##
# Augmentations affecting intensity image only (Motion, Gaussian Blur, Cut-out)
##
class TorchivisionAug(object):
    def __init__(self, cfg):
        self.transforms = []
        self.all_augs = {"GaussianBlur": self.add_gaussian_blur(),
                         "ZeroPad": self.add_zeropad(cfg.DATA.PADDED_SIZE)
                         }

    def get_transforms(self):
        return self.transforms

    def add_zeropad(self, padding_size):
        self.transforms.append(ZeroPad2D((padding_size, padding_size)))

    def add_gaussian_blur(self):
        tt.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))

    def add_motion(self):
        pass

    def add_cut_out(self):
        pass

    def add_bf(self):
        pass

    def add_augs_from_list(self, augs):
        for aug in augs:
            self.transforms.append(self.all_augs[aug])


##
# Random affine effecting labels (Rotation, Translation Scaling from original image)
##
class RandomRotation2D(object):
    def __init__(self, angle):
        """
         Pad the input with zeros to get output size
        :param output_size:
        :param pos: position to put the input
        """
        assert isinstance(angle, (int, float)), "Argument angle should be int or float"
        self.degrees = tt.functional._setup_angle(angle, name="degrees", req_size=(2,))

    def _random_affine(self, image, center=None):
        """
        Affine matrix is : M = T * C * RotateScaleShear * C^-1
        Inverse is : M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1
        :param image:
        :param center:
        :return:
        """
        # Get random Rotation Angles (within specified range)
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())

        if len(image.shape) == 2:
            h, w = image.shape
        else:
            h, w, c = image.shape

        if center is not None:
            # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
            center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [w, h])]
        else:
            center_f = [0.0, 0.0]

        # ResampleImageFilter expects the transform from the output space to
        # the input space. Intuitively, the passed arguments should take us
        # from the input space to the output space, so we need to invert the
        # transform.
        # More info at https://github.com/fepegar/torchio/discussions/693
        matrix = tt.functional._get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])

        return matrix

    def __call__(self, sample):

        sample["affine"] = self._random_affine(sample['img'])

        return sample


##
# Transformations for evaluation
##
class ToTensorTest(ToTensor):
    """
    Convert np.ndarrays in sample to Tensors.
    """

    def __init__(self, dim=2):
        super(ToTensorTest, self).__init__(dim)

    def __call__(self, img):
        img = super(ToTensorTest, self).norm_and_transpose(img)
        img = torch.from_numpy(img)
        return img


class ZeroPadTest(ZeroPad):
    def __init__(self, output_size, pos='top_left', dim=2):
        """
         Pad the input with zeros to get output size
        :param output_size:
        :param pos: position to put the input
        """
        super(ZeroPadTest, self).__init__(output_size, pos, dim)

    def __call__(self, img):
        img = super(ZeroPadTest, self)._pad(img)

        return img
