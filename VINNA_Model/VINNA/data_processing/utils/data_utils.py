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
import pandas as pd
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion, binary_closing, filters, uniform_filter, affine_transform
import nibabel as nib
import torch
import re
import math


##
# Helper Functions
##
MAPPINGS = {
    1: 17,
    17: [8, 7],
    7: 2,
    2: 53,
    5: 3,
    3: 18,
    49: 4,
    4: 54,
    6: 42,
    8: 41,
    9: 9,
    10: 48,
    18: [46, 47],
    19: 16,
    40: 11,
    41: 50,
    50: 43,
    48: [251, 252, 253, 254, 255],
    83: 24
}

ROT_TL_INDICES = {"coronal": [-1, 0, 1],
                  "axial": [1, 0, 1],
                  "sagittal": [0, 1, 2]}


def mapping_fs_to_dhcp(img, mapping=MAPPINGS):
    """
    Mapping 32 FreeSurfer labels to dHCP labels
    :param np.ndarray img: image to map
    :return np.ndarray: mapped/reduced image
    """
    for new_lab, lab in mapping.items():
        mask = np.in1d(img, lab).reshape(img.shape)
        img[mask] = new_lab
    return img


def crop_image_from_affine(large_img, small_img, save_as=None, dtype=None):
    """
    Crop a larger image with identical zoom and rotation to match a smaller one.
    Information is taken from the affine multiplied with the ras-zero coordinate and
    the shape of the smaller image. If the save_as parameter is specified, the image is saved
    to the defined file.

    Args:
        nib.Nifti1Image/nib.MGHImage/str large_img: large image to crop (full information with affine, header and data)
        nib.Nifti1Image/nib.MGHImage/str small_img: small image to crop to (full information with affine, header and data)
        save_as: name under which to save prediction; this determines output file format
        dtype: image array type; if provided, the image object is explicitly set to match this type

    Returns: nothing/None, saves predictions to save_as
    """
    if isinstance(large_img, str):
        large_img = nib.load(large_img)
    if isinstance(small_img, str):
        small_img = nib.load(small_img)
    ras_zero = np.asarray([0, 0, 0, 1])
    ras_zero_vox_min = np.linalg.inv(small_img.affine) @ ras_zero
    ras_zero_vox_max = np.linalg.inv(large_img.affine) @ ras_zero
    start = (ras_zero_vox_max[:-1] - ras_zero_vox_min[:-1]).astype(int)
    stop = start + small_img.shape
    large_cut = np.asanyarray(large_img.dataobj)[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    if save_as is not None:
        save_image_as_nifti(small_img.header, small_img.affine, large_cut, save_as, dtype_set=dtype)
    return large_cut


# Save image routine
def save_image_as_nifti(header, affine, image_to_save, save_as, dtype_set=np.int16):
    """
    Function to save a given file as a nifti-image
    :param string original_image: name and directory of the original image
    :param ndarray image_to_save: image with dimensions (height, width, depth) to be saved in nifti format
    :param save_as: name and directory where the nifti should be stored
    :return: void
    """
    # Generate new nifti file
    if save_as.endswith("mgz"):
        mgh_img = nib.MGHImage(image_to_save, affine, header)
    elif any(save_as.endswith(file_ext) for file_ext in ["nii", "nii.gz"]):
        mgh_img = nib.nifti1.Nifti1Pair(image_to_save, affine, header)
    else:
        raise Exception(f"File Ending is not supported {save_as}")

    if dtype_set is not None:
        mgh_img.set_data_dtype(np.dtype(dtype_set))  # not uint8 if aparc!!! (only goes till 255)

    if any(save_as.endswith(file_ext) for file_ext in ["mgz", "nii"]):
        nib.save(mgh_img, save_as)
    elif save_as.endswith("nii.gz"):
        # For correct outputs, nii.gz files should be saved using the nifti1 sub-module's save():
        nib.nifti1.save(mgh_img, save_as)
    else:
        raise Exception(f"File Ending is not supported {save_as}")


# Transformation for mapping
def transform_axial(vol, coronal2axial=True):
    """
    Function to transform volume into Axial axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2axial: transform from coronal to axial = True (default),
                               transform from axial to coronal = False
    :return:
    """
    if coronal2axial:
        return np.moveaxis(vol, [0, 1, 2], [1, 2, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 0, 1])


def transform_sagittal(vol, coronal2sagittal=True):
    """
    Function to transform volume into Sagittal axis and back
    :param np.ndarray vol: image volume to transform
    :param bool coronal2sagittal: transform from coronal to sagittal = True (default),
                                transform from sagittal to coronal = False
    :return:
    """
    if coronal2sagittal:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])
    else:
        return np.moveaxis(vol, [0, 1, 2], [2, 1, 0])


# Thick slice generator (for eval) and blank slices filter (for training)
def get_thick_slices(img_data, slice_thickness=3):
    """
    Function to extract thick slices from the image
    (feed slice_thickness preceeding and suceeding slices to network,
    label only middle one)
    :param np.ndarray img_data: 3D MRI image read in with nibabel
    :param int slice_thickness: number of slices to stack on top and below slice of interest (default=3)
    :return:
    """
    h, w, d = img_data.shape
    img_data_pad = np.expand_dims(np.pad(img_data, ((slice_thickness, slice_thickness), (0, 0), (0, 0)), mode='edge'),
                                  axis=3)
    img_data_thick = np.ndarray((h, w, d, 0), dtype=np.uint8)

    for slice_idx in range(2 * slice_thickness + 1):
        img_data_thick = np.append(img_data_thick, img_data_pad[slice_idx:d + slice_idx, ...], axis=3)

    return img_data_thick


def filter_blank_slices_thick(volumes, threshold=50):
    """
    Function to filter blank slices from the volume using the label volume
    :param np.ndarray img_vol: orig image volume
    :param np.ndarray label_vol: label images (ground truth)
    :param np.ndarray weight_vol: weight corresponding to labels
    :param int threshold: threshold for number of pixels needed to keep slice (below = dropped)
    :return:
    """
    # Get indices of all slices with more than threshold labels/pixels
    select_slices = (np.sum(volumes["Labels"], axis=(0, 1)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    for name, img in volumes.items():
        if len(img.shape) == 4:
            volumes[name] = img[:, :, select_slices, :]
        else:
            volumes[name] = img[:, :, select_slices]

    return volumes


def filter_blank_volumes(volumes, threshold=50):
    # Get indices of all slices with more than threshold labels/pixels
    select_slices_z = (np.sum(volumes["Labels"], axis=(0, 1)) > threshold)
    select_slices_x = (np.sum(volumes["Labels"], axis=(1, 2)) > threshold)
    select_slices_y = (np.sum(volumes["Labels"], axis=(0, 2)) > threshold)

    # Retain only slices with more than threshold labels/pixels
    for name, img in volumes.items():
        print(f"Shape of {name} before {img.shape}, after {img[select_slices_x, select_slices_y, select_slices_z].shape}")
        volumes[name] = img[select_slices_x, select_slices_y, select_slices_z]

    return volumes

# Cropping/Padding functions
def bounding_box_slices(label, size_half=128):
    """
    Function to get slices for cropping an image to a given bounding box (avoid cropping of brain tissue)
    :param np.ndarray label: ground truth image
    :param int size_half: half size of final desired image dimension (default=128 --> image 256x256x256)
    :return:
    """
    boundary_point = np.asarray(label.shape) - size_half

    # 1. reduce rows and cols with non-zero values to 1D vector
    height = np.any(label, axis=(1, 2))
    width = np.any(label, axis=(0, 2))
    depth = np.any(label, axis=(0, 1))

    # 2. Find indices with np.where, take first and last non-zero index for bbox
    height_min, height_max = np.where(height)[0][[0, -1]]
    width_min, width_max = np.where(width)[0][[0, -1]]
    depth_min, depth_max = np.where(depth)[0][[0, -1]]

    # 3. Get middle value --> starting point
    middle_point = np.asarray(list(map(lambda val_max, val_min: int(np.rint((val_max + val_min) / 2)),
                                       [height_min, width_min, depth_min], [height_max, width_max, depth_max])))
    final_mid = np.asarray((list(map(lambda middle, boundary: min(max(middle, size_half), boundary), middle_point,
                                     boundary_point))))
    start_slice = final_mid - size_half
    stop_slice = final_mid + size_half

    return np.floor(start_slice).astype(int), np.floor(stop_slice).astype(int)


def bounding_box_crop(images, slices_start, slices_stop):
    """
    Function to crop image, label and optionally weights to a given bounding box
    :param dict(str: np.ndarray) images: dict of images to crop
    :param list(int) slices_start: start coordinates of bounding box
    :param list(int) slices_stop: stop coordinates of bounding box
    :return:
    """
    for image in images.keys():
        if images[image] is not None:
            images[image] = images[image][slices_start[0]:slices_stop[0], slices_start[1]:slices_stop[1], slices_start[2]:slices_stop[2]]
    return images


def bounding_box_pad(image, slices_start, slices_stop, size):
    """
    Function to pad image to a given size (zero padding)
    :param np.ndarray image: image to pad
    :param list(int) slices_start: start coordinates of original bounding box
    :param list(int) slices_stop: stop coordinates of original bounding box
    :param tuple(int) size: final output size of padded image
    :return:
    """
    new_sized_image = np.zeros(shape=size, dtype=np.int16)
    if len(size) > 3:
        image = np.transpose(image, (3, 0, 1, 2))
        new_sized_image[:, slices_start[0]:slices_stop[0], slices_start[1]:slices_stop[1],
        slices_start[2]:slices_stop[2]] = image
        new_sized_image = np.transpose(new_sized_image, (1, 2, 3, 0))
    else:
        new_sized_image[slices_start[0]:slices_stop[0], slices_start[1]:slices_stop[1],
        slices_start[2]:slices_stop[2]] = image
    return new_sized_image


# weight map generator
def create_weight_mask(mapped_aseg, labels, ctx_labels, ext, max_weight=5, max_edge_weight=5, max_hires_weight=None,
                       mean_filter=False, use_cortex_mask=True, gradient=True):
    """
    Function to create weighted mask - with median frequency balancing and edge-weighting
    :param mapped_aseg:
    :param max_weight:
    :param max_edge_weight:
    :return:
    """
    class_count = np.zeros(shape=labels.shape)
    unique, counts = np.unique(mapped_aseg, return_counts=True)
    class_count[unique] = counts

    # Median Frequency Balancing
    class_wise_weights = np.median(counts) / class_count
    class_wise_weights[class_wise_weights > max_weight] = max_weight
    (h, w, d) = mapped_aseg.shape
    weights_mask = np.reshape(class_wise_weights[mapped_aseg.ravel()], (h, w, d))

    # Gradient Weighting
    if gradient:
        (gx, gy, gz) = np.gradient(mapped_aseg)
        grad_weight = max_edge_weight * np.asarray(np.power(np.power(gx, 2) + np.power(gy, 2) + np.power(gz, 2), 0.5) > 0,
                                                   dtype='float')

        weights_mask += grad_weight

    if max_hires_weight is not None:
        # High-res Weighting
        print("Adding hires weight mask deep sulci and WM with weight ", max_hires_weight)
        ctx_mask = np.in1d(mapped_aseg, ctx_labels).reshape(mapped_aseg.shape)
        mask1 = deep_sulci_and_wm_strand_mask(mapped_aseg, structure=np.ones((3, 3, 3)), ctx_mask=ctx_mask)
        weights_mask += mask1 * max_hires_weight

        if use_cortex_mask:
            print("Adding cortex mask with weight ", max_hires_weight)
            #non_ctx_mask = (label <= ctx_thresh) = ~ctx_mask
            mask2 = cortex_border_mask(mapped_aseg, structure=np.ones((3, 3, 3)), ctx_mask=ctx_mask, ext_labels=ext)
            weights_mask += mask2 * (max_hires_weight) // 2

    if mean_filter:
        weights_mask = uniform_filter(weights_mask, size=3)

    return weights_mask


def cortex_border_mask(volume, structure, ctx_mask, ext_labels):
    """
    Function to erode the cortex of a given mri image to create
    the inner gray matter mask (outer most cortex voxels)
    :param np.ndarray label: ground truth labels.
    :param np.ndarray structure: structuring element to erode with
    :return: np.ndarray outer GM layer
    """
    # create aseg brainmask, erode it and subtract from itself
    external_mask = np.in1d(volume, ext_labels).reshape(volume.shape)
    bm = np.zeros(shape=volume.shape)
    bm[~external_mask] = 1

    eroded = binary_erosion(bm, structure=structure)
    diff_im = np.logical_xor(eroded, bm)

    # only keep values associated with the cortex
    diff_im[~ctx_mask] = 0  # > 33 (>19) = > 1002 in FS space (full (sag)),
    print("Remaining voxels cortex border: ", np.unique(diff_im, return_counts=True))
    return diff_im


def deep_sulci_and_wm_strand_mask(volume, structure, ctx_mask, iteration=1):
    """
    Function to get a binary mask of deep sulci and small white matter strands
     by using binary closing (erosion and dilation)

    :param np.ndarray volume: loaded image (aseg, label space)
    :param np.ndarray structure: structuring element (e.g. np.ones((3, 3, 3)))
    :param int iteration: number of times mask should be dilated + eroded (default=1)
    :return np.ndarray: sulcus + wm mask
    """
    # Binarize label image (cortex = 1, everything else = 0)
    empty_im = np.zeros(shape=volume.shape)
    empty_im[ctx_mask] = 1  # > 33 (>19) = >1002 in FS LUT (full (sag))

    # Erode the image
    eroded = binary_closing(empty_im, iterations=iteration, structure=structure)

    # Get difference between eroded and original image
    diff_image = np.logical_xor(empty_im, eroded)
    print("Remaining voxels sulci/wm strand: ", np.unique(diff_image, return_counts=True))
    return diff_image


# Label mapping functions (to aparc (eval) and to label (train))
def read_classes_from_lut(lut_file):
    """
    Function to read in FreeSurfer-like LUT table
    :param str lut_file: path and name of FreeSurfer-style LUT file
                         Example entry:
                         ID LabelName  R   G   B   A
                         0   Unknown   0   0   0   0
                         1   Left-Cerebral-Exterior 70  130 180 0
    :return pd.Dataframe: DataFrame with ids present, name of ids, color for plotting
    """
    # Read in file
    separator = {"tsv": "\t", "csv": ",", "txt": " "}
    return pd.read_csv(lut_file, sep=separator[str(lut_file)[-3:]])


def map_label2aparc_aseg(mapped_aseg, labels):
    """
    Function to perform look-up table mapping from sequential label space to LUT space
    :param np.ndarray mapped_aseg: label space segmentation (aparc.DKTatlas + aseg)
    :param list(int) labels: list of labels defining LUT space
    :return:
    """
    h, w, d = mapped_aseg.shape
    #assert set(torch.unique(mapped_aseg)).issubset(range(0, len(labels))), "Error: more aseg labels than expected. Got \n{}\n{}".format(np.unique(mapped_aseg), list(range(len(labels))))
    if torch.is_tensor(mapped_aseg):
        aseg = labels[torch.ravel(mapped_aseg)]
        aseg = torch.reshape(aseg, (h, w, d))
    else:
        aseg = labels[np.ravel(mapped_aseg)]
        aseg = np.reshape(aseg, (h, w, d))

    return aseg


def clean_labels(aseg):
    aseg[aseg == 80] = 77  # Hypointensities Class
    aseg[aseg == 85] = 0  # Optic Chiasma to BKG
    aseg[aseg == 62] = 41  # Right Vessel to Right WM
    aseg[aseg == 30] = 2  # Left Vessel to Left WM
    aseg[aseg == 72] = 24  # 5th Ventricle to CSF
    aseg[aseg == 29] = 0  # left-undetermined to 0
    aseg[aseg == 61] = 0  # right-undetermined to 0
    aseg[(aseg >= 251) & (aseg <= 255)] = 2 # map cc to WM
    aseg[aseg == 1000] = 3  # Map unknown to cortex
    aseg[aseg == 2000] = 42
    return aseg


def map_aparc_aseg2label(aseg, labels, aseg_nocc=None, mapping=True):
    """
    Function to perform look-up table mapping of aparc.DKTatlas+aseg.mgz data to label space
    :param np.ndarray aseg: ground truth aparc+aseg
    :param None/np.ndarray aseg_nocc: ground truth aseg without corpus callosum segmentation
    :return:
    """
    # If corpus callosum is not removed yet, do it now
    if aseg_nocc is not None:
        cc_mask = (aseg >= 251) & (aseg <= 255)
        aseg[cc_mask] = aseg_nocc[cc_mask]

    aseg_temp = aseg.copy()
    if mapping:
        aseg[aseg == 80] = 77  # Hypointensities Class
        aseg[aseg == 62] = 41  # Right Vessel to Right GM
        aseg[aseg == 30] = 2  # Left Vessel to Left GM
        aseg[aseg == 72] = 24  # 5th Ventricle to CSF
        aseg[aseg_temp == 1000] = 3  # Map unknown to cortex
        aseg[aseg_temp == 2000] = 42

    assert set(np.unique(aseg)).issubset(labels), "Error: aseg classes not subset of labels: \n{}\n{}".format(np.unique(aseg), labels)
    assert not np.any(aseg > 175), "Error: classes above 175 still exist in aseg {}".format(np.unique(aseg))
    assert np.any(aseg == 3) and np.any(aseg == 42), "Error: no cortical marker detected {}".format(np.unique(aseg))

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels) + 1, dtype='int')
    for idx, value in enumerate(labels):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial

    mapped_aseg = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg = mapped_aseg.reshape((h, w, d))

    # Map Sagittal Labels
    aseg[aseg == 2] = 41
    aseg[aseg == 3] = 42
    aseg[aseg == 4] = 43
    aseg[aseg == 5] = 44
    aseg[aseg == 7] = 46
    aseg[aseg == 8] = 47
    aseg[aseg == 10] = 49
    aseg[aseg == 11] = 50
    aseg[aseg == 12] = 51
    aseg[aseg == 13] = 52
    aseg[aseg == 17] = 53
    aseg[aseg == 18] = 54
    aseg[aseg == 26] = 58
    aseg[aseg == 28] = 60
    aseg[aseg == 31] = 63

    labels_sag = np.array([0, 14, 15, 41, 42, 43, 46, 47, 49,
                           50, 51, 52, 53, 54, 58, 60, 85, 172, 173, 174, 175])

    h, w, d = aseg.shape
    lut_aseg = np.zeros(max(labels_sag) + 1, dtype='int')
    for idx, value in enumerate(labels_sag):
        lut_aseg[value] = idx

    # Remap Label Classes - Perform LUT Mapping - Coronal, Axial

    mapped_aseg_sag = lut_aseg.ravel()[aseg.ravel()]

    mapped_aseg_sag = mapped_aseg_sag.reshape((h, w, d))

    return mapped_aseg, mapped_aseg_sag


def sagittal_coronal_remap_lookup(x):
    """
    Dictionary mapping to convert left labels to corresponding right labels for aseg
    :param int x: label to look up
    :return:
    """
    return {
        2: 41,
        3: 42,
        4: 43,
        7: 46,
        8: 47,
        10: 49,
        11: 50,
        12: 51,
        13: 52,
        17: 53,
        18: 54,
        26: 58,
        28: 60
    }[x]


def unify_lateralized_labels(lut, combi=["Left-", "Right-"]):
    """
    Function to generate lookup dictionary of left-right labels
    :param str or pd.DataFrame lut: either lut-file string to load or pandas dataframe
                                    Example entry:
                                    ID LabelName  R   G   B   A
                                    0   Unknown   0   0   0   0
                                    1   Left-Cerebral-Exterior 70  130 180 0
    :param list(str) combi: Prefix or labelnames to combine. Default: Left- and Right-
    :return dict: dictionary mapping between left and right hemispheres
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    left = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[0])]
    right = lut[["ID", "LabelName"]][lut["LabelName"].str.startswith(combi[1])]
    left["LabelName"] = left["LabelName"].str.removeprefix(combi[0])
    right["LabelName"] = right["LabelName"].str.removeprefix(combi[1])
    mapp = left.merge(right, on="LabelName")
    return pd.Series(mapp.ID_y.values, index=mapp.ID_x).to_dict()


def get_labels_from_lut(lut, label_extract=("Left-", "ctx-rh")):
    """
    Function to extract
    :param str of pd.DataFrame lut: FreeSurfer like LookUp Table (either path to it
                                    or already loaded as pandas DataFrame.
                                    Example entry:
                                    ID LabelName  R   G   B   A
                                    0   Unknown   0   0   0   0
                                    1   Left-Cerebral-Exterior 70  130 180 0
    :param tuple(str) label_extract: suffix of label names to mask for sagittal labels
                                     Default: "Left-" and "ctx-rh"
    :return np.ndarray: full label list
    :return np.ndarray: sagittal label list
    """
    if isinstance(lut, str):
        lut = read_classes_from_lut(lut)
    mask = lut["LabelName"].str.startswith(label_extract)
    return lut["ID"].values, lut["ID"][~mask].values


def infer_mapping_from_lut(lut):
    labels, labels_sag = get_labels_from_lut(lut)
    sagittal_mapping = unify_lateralized_labels(lut)
    idx_list = np.ndarray(shape=(labels.shape[0],), dtype=np.int16)
    for idx in range(len(labels)):
        idx_in_sag = np.where(labels_sag == labels[idx])[0]
        if idx_in_sag.size == 0:
            current_label_sag = sagittal_mapping[labels[idx]]
            idx_in_sag = np.where(labels_sag == current_label_sag)[0]

        idx_list[idx] = idx_in_sag
    return idx_list


def map_prediction_sagittal2full(prediction_sag, lut=None):
    """
    Function to remap the prediction on the sagittal network to full label space used by coronal and axial networks
    (full aparc.DKTatlas+aseg.mgz)
    :param prediction_sag: sagittal prediction (labels)
    :param int num_classes: number of classes (96 for full classes, 79 for hemi split, 36 for aseg)
    :return: Remapped prediction
    """
    assert lut is not None, 'lut is not defined!'
    idx_list = infer_mapping_from_lut(lut)
    prediction_full = prediction_sag[:, idx_list, :, :]
    return prediction_full


# Clean up and class separation
def bbox_3d(img):
    """
    Function to extract the three-dimensional bounding box coordinates.
    :param np.ndarray img: mri image
    :return:
    """

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def get_largest_cc(segmentation):
    """
    Function to find largest connected component of segmentation.
    :param np.ndarray segmentation: segmentation
    :return:
    """
    labels = label(segmentation, connectivity=3, background=0)

    bincount = np.bincount(labels.flat)
    background = np.argmax(bincount)
    bincount[background] = -1

    largest_cc = labels == np.argmax(bincount)

    return largest_cc


def get_coordinate_grid_from_image(image):
    """
    Function to create 3D grid from image. Resulting shape is (N, 3) with N = dim1 x dim2 x dim3.
    :param np.ndarray image: image data to create grid from. Shape has to be HxWxD (no channel)
    :return np.ndarray: coordinate grid for given image
    """
    assert len(image.shape) == 3
    # Create coordinates (N, 3)
    dim = np.array(image.shape)
    x, y, z = np.meshgrid(np.arange(dim[0]), np.arange(dim[1]), np.arange(dim[2]), indexing="ij")
    return np.array([x.flatten(), y.flatten(), z.flatten()]).transpose()


def lta_to_affine(lta_infos, pre=""):
    """
    Create correct affine for an image from lta infos.
    :param dict lta_infos: dict with all necessary infos:
                           - list(float) xras: [x_r, x_a, x_s] x-RAS location (=x-axis origin in RAS location)
                           - list(float) yras: [y_r, y_a, y_s] y-RAS location (=y-axis origin in RAS location)
                           - list(float) zras: [z_r, z_a, z_s] z-RAS location (=z-axis origin in RAS location)
                           - list(float) cras: [c_r, c_a, c_s] c-RAS location (=central voxel of the image in RAS)
                           - list(int) voxelsize: voxel size (resolution) of the image --> [x-size, y-size, z-size] for 3D
                           - list(int) volume: volume shape of the image --> [width, height, depth] for 3D
    :param str pre: pre-fix to add to keys (e.g. src_ or dst_ for source and destination affine)
    :return np.ndarray: reconstructed affine vox2ras matrix
    """
    # Construct Rotation/Scaling/Shearing part from xras, yras, zras and voxelsize
    A = np.concatenate((np.c_[lta_infos[pre + 'xras']], np.c_[lta_infos[pre + 'yras']], np.c_[lta_infos[pre + 'zras']]),
                       axis=1) * lta_infos[pre + 'voxelsize']

    # Construct Translation part from cras, A and volume info (account for scaling/shear and image size to get
    # correct central voxel position)
    b = lta_infos[pre + 'cras'] - A @ lta_infos[pre + 'volume'] / 2

    # Combine everything into an affine matrix
    nin, nout = A.shape
    affine = np.eye(nin + 1)
    affine[0:nin, 0:nout] = A
    affine[0:nin, nout] = b
    return affine

def read_lta(file):
    """
    Function to read a linear transform array (FreeSurfer make_upright output) into a dict.
    Final dict contains information about the source image, target image and affine to register
    between both of them.

    Please note, lta uses the "points" convention:
        - the source is the reference (coordinates for which we need to find a projection)
        - the destination is the moving image (from which data is pulled-back).
    Hence, the matrix of an LTA is defined such that it maps coordinates from the output space (dst volume) to the
    input space (src volume) = pull or backward resampling, same as for scipy/numpy.ndimage.affine_transform.
    Therefore, without inversion, the LTA matrix is appropiate to move the information from input space (src volume)
    into the output space (dest volume's) grid.

    Affine transformations are often described in the ‘push’ (or ‘forward’) direction, transforming input to output.
    If you want a matrix for the ‘push’ transformation, use the inverse (numpy.linalg.inv) of the returned lta-affine.

    :param str file: path to lta-file
    :return dict: dictionary with affine and header info of source and target image
    """
    # Open the files and read it into file
    with open(file, 'r') as f:
        lta = f.readlines()
    d = dict()
    i = 0
    # Loop over the lines and extract information for type (vox2vox = 0, ras2ras = 1), nxforms, mean and sigma of affine
    # the actual lta-affine (stored in ["lta"]) as well as the source and target header information.
    while i < len(lta):
        # Read lta-affine info
        if re.match('type', lta[i]) is not None:
            d['type'] = int(re.sub('=', '', re.sub('[a-z]+', '', re.sub('#.*', '', lta[i]))).strip())
            i += 1
        elif re.match('nxforms', lta[i]) is not None:
            d['nxforms'] = int(re.sub('=', '', re.sub('[a-z]+', '', re.sub('#.*', '', lta[i]))).strip())
            i += 1
        elif re.match('mean', lta[i]) is not None:
            d['mean'] = [float(x) for x in
                         re.split(" +", re.sub('=', '', re.sub('[a-z]+', '', re.sub('#.*', '', lta[i]))).strip())]
            i += 1
        elif re.match('sigma', lta[i]) is not None:
            d['sigma'] = float(re.sub('=', '', re.sub('[a-z]+', '', re.sub('#.*', '', lta[i]))).strip())
            i += 1
        elif re.match('-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+', lta[i]) is not None:
            d['lta'] = np.array([
                [float(x) for x in re.split(" +",
                                            re.match('-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+',
                                                     lta[i]).string.strip())],
                [float(x) for x in re.split(" +",
                                            re.match('-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+',
                                                     lta[i + 1]).string.strip())],
                [float(x) for x in re.split(" +",
                                            re.match('-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+',
                                                     lta[i + 2]).string.strip())],
                [float(x) for x in re.split(" +",
                                            re.match('-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+-*[0-9]\.\S+\W+',
                                                     lta[i + 3]).string.strip())]
            ])
            i += 4
        # Read source volume header information
        elif re.match('src volume info', lta[i]) is not None:
            while i < len(lta) and re.match('dst volume info', lta[i]) is None:
                if re.match('valid', lta[i]) is not None:
                    d['src_valid'] = int(re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())
                elif re.match('filename', lta[i]) is not None:
                    d['src_filename'] = re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())
                elif re.match('volume', lta[i]) is not None:
                    d['src_volume'] = [int(x) for x in
                                       re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('voxelsize', lta[i]) is not None:
                    d['src_voxelsize'] = [float(x) for x in
                                          re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('xras', lta[i]) is not None:
                    d['src_xras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('yras', lta[i]) is not None:
                    d['src_yras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('zras', lta[i]) is not None:
                    d['src_zras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('cras', lta[i]) is not None:
                    d['src_cras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                i += 1
        # Read target volume header information
        elif re.match('dst volume info', lta[i]) is not None:
            while i < len(lta) and re.match('src volume info', lta[i]) is None:
                if re.match('valid', lta[i]) is not None:
                    d['dst_valid'] = int(re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())
                elif re.match('filename', lta[i]) is not None:
                    d['dst_filename'] = re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())
                elif re.match('volume', lta[i]) is not None:
                    d['dst_volume'] = [int(x) for x in
                                       re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('voxelsize', lta[i]) is not None:
                    d['dst_voxelsize'] = [float(x) for x in
                                          re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('xras', lta[i]) is not None:
                    d['dst_xras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('yras', lta[i]) is not None:
                    d['dst_yras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('zras', lta[i]) is not None:
                    d['dst_zras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                elif re.match('cras', lta[i]) is not None:
                    d['dst_cras'] = [float(x) for x in
                                     re.split(" +", re.sub('.*=', '', re.sub('#.*', '', lta[i])).strip())]
                i += 1
        else:
            i += 1
    # create full transformation matrices for source and target image
    d['src'] = lta_to_affine(d, pre="src_")
    d['dst'] = lta_to_affine(d, pre="dst_")

    if d['type'] == 1:
        # lta is in Ras2Ras affine --> vox2ras @ ras2ras @ ras2vox
        d['lta_vox2vox'] = np.matmul(np.linalg.inv(d['dst']), np.matmul(d['lta'], d['src']))
        d['lta_ras2ras'] = d['lta']
    elif d['type'] == 0:
        # lta is already in Vox2Vox, get ras2ras --> ras2vox @ vox2vox @ vox2ras
        d['lta_vox2vox'] = d['lta']
        d['lta_ras2ras'] = np.matmul(d['dst'], np.matmul(d['lta_vox2vox'], np.linalg.inv(d['src'])))

    # return info dict
    return d


def is_rotation_matrix(r, epsilon=1e-6):
    """
    Check if inserted matrix is valid rotation matrix (i.e. no
    extra shear or scaling present). This is the case if the determinant is 1 and
    R (R^T) = (R^T) M = I. We check with determinant 1 and identity to some error (1e-6)
    :param np.ndarray r: rotation matrix to check for properness
    :param float epsilon: value below which (R^T) M = I is true and determinant is 1; default: 1e-6
    :return bool: True, if rotation matrix is proper, else False
    """
    r_t = np.transpose(r)
    should_be_identity = np.dot(r_t, r)
    i = np.identity(r.shape[0], dtype=r.dtype)
    n = (np.linalg.norm(i - should_be_identity) < epsilon) and (1 + epsilon > np.linalg.det(r) > 1-epsilon)
    return n


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_to_euler(r, angles=False):
    """
    Extract euler angles from rotation matrix. Order is x, y, z. If angles is set
    to True, the result is returned in degrees, otherwise as radiants.

    :param np.ndarray r: rotation matrix. Needs to be a valid rotation matrix (i.e. no shear
                         or scaling can be included.
    :param bool angles: Flag to indicate return type of angles (degres or radiants)
    :return np.ndarray: x, y, z euler angles
    """
    assert (is_rotation_matrix(r))

    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0
    return np.array([x, y, z]) if not angles else np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


def decompose_tsr(transformation_matrix, angles=False):
    """
    Get rotation, scaling and translation from 2D transformation matrix.
    The matrix is represented as [[a, b, e],
                                  [c, d, f]].
    Matrix must be in radiants (normalized -1 to 1). If result should be returned as
    degrees (0-360), set angles to True. Rotations are given out as clock-wise about the origin.

    The two-dimensional case is the only non-trivial (i.e. not one-dimensional) case where
    the rotation matrices group is commutative, so that it does not matter in which order
    multiple rotations are performed.

    :param np.ndarray transformation_matrix: 2D transformation matrix to decompose
    :param bool angles: return rotation results in degrees not radiants
    :return dict: dictionary with x, y translation, scaling and rotation angle
    """
    import math
    a, b, e, c, d, f = transformation_matrix[0:2, 0:3].flatten()
    if a != 0 or c != 0:
        hypotac = math.hypot(a, c)
        scale_x = hypotac
        scale_y = (a * d - b * c) / hypotac
        acos = math.acos(a / hypotac)
        rotation = -acos if c < 0 else acos
    elif b != 0 or d != 0:
        hypotbd = math.hypot(b, d)
        scale_x = (a * d - b * c) / hypotbd
        scale_y = hypotbd
        acos = math.acos(b / hypotbd)
        rotation = math.pi / 2 + (-acos if d < 0 else acos)
    else:
        scale_x = 0
        scale_y = 0
        rotation = 0
    return {"translate": {"tx": e, "ty": f},
            "scale": {"sx": scale_x, "sy": scale_y},
            "rotation": math.degrees(rotation) if angles else rotation}


def new_identity(shape, eye_shape, type_hint):
    """
    Create identiy matrix for given shape
    Args:
        tuple(int) shape: final shape identity shoule be formed into
        tuple(int) eye_shape: shape of identiy matrix to create
        np.ndarray type_hint: array with dtype that identity should have
    Returns:
       Identity matrix with given shape.
    """
    eye = np.eye(*eye_shape, dtype=type_hint.dtype)
    eye = eye.reshape((1,) * len(shape) + eye.shape)
    eye = np.tile(eye, tuple(shape) + (1, 1))
    return eye


def diagonal(matrix, offset=0):
    """
    View of the diagonal along the last two axis offset by offset. Returns object of reduced shape.
    Args:
         np.ndarray matrix: Matrix to extract diagonal from
         int offset: offset to apply, if any (default=0)
    Returns:
         Diagonal along the last two axis offset by offset
    """
    if offset == 0:
        sub = matrix
    elif offset > 0:
        sub = matrix[..., offset:, :min(matrix.shape[-2] - offset, matrix.shape[-1])]
    else:
        sub = matrix[..., :min(matrix.shape[-1] + offset, matrix.shape[-2]), -offset:]
    reshaped_view = sub.reshape(matrix.shape[:-2] + (-1,))
    return reshaped_view[..., ::sub.shape[-1] + 1]


def convert_affine(matrix, target_format, source_format=None):
    """
    Converts the affine matrix assuming one format into a different target format.
    Args:
        np.ndarray/torch.tensor matrix: source affine matrix according to `source_format`
        str target_format: format to convert to from '(n+1)x(n+1)', 'nx(n+1)', 'nxn', 'n'.
        str source_format: format of the source from '(n+1)x(n+1)', 'nx(n+1)', 'nxn', 'n', will be inferred by
            infer_affine_format, if source_format is not passed.
    Returns:
        The matrix of affine matrices converted to the
    Raises:
        TypeError: If target_format or source_format are not strings or not valid AffineFormatSpecifiers.
    """

    ndim = matrix.shape[-1] - (0 if source_format.endswith('n') else 1)
    shape = matrix.shape[:-1] if source_format == 'n' else matrix.shape[:-2]

    if target_format == source_format:
        return matrix

    formats = ["(n+1)x(n+1)", "nx(n+1)", "nxn", "n"]
    source_id = formats.index(source_format)
    target_id = formats.index(target_format)

    shape_aff = (ndim + (0 if target_format.startswith('n') else 1),)
    if target_format != 'n':
        shape_aff = shape_aff + (ndim + (0 if target_format.endswith('n') else 1),)

    if source_id > target_id:
        # "larger" affine matrix, always works; always numpy
        data = new_identity(shape, shape_aff, type_hint=matrix)
        if source_format == 'n':
            diagonal_view = diagonal(data)
            diagonal_view[..., :ndim] = matrix
        else:
            data[..., :matrix.shape[-2], :matrix.shape[-1]] = matrix
        if target_format == '(n+1)x(n+1)':
            data[..., -1, -1] = 1
        return data
    else:
        e_vec = np.zeros((1,) * len(shape) + (ndim + 1,), matrix.dtype)
        e_vec[..., -1] = 1
        #logger.info(f"'smaller' affine matrix, prone to discarded data! Some steps might get discarded!")

        if target_format == 'n':
            # maybe warn
            return diagonal(matrix)[..., :shape_aff[0]]
        else:
            return matrix[..., :shape_aff[0], :shape_aff[1]]


def normalize_affine(affine, source_shape, target_shape=None, affine_format="nx(n+1)"):
    """
    Converts an affine from an unnormalized style (numpy/scipy) to a normalized style (torch).
    Args:
        affine np.ndarray/torch.tensor: The (batch of) affine matrices
        source_shape tuple(int): The shape of the image, so the center can be determined.
        target_shape (optional) tuple(int): The shape of the target image, so the correct transformation can be determined
            (defaults to source_shape).
        affine_format (optional) str: The target affine format (default: "nx(n+1)")
    Returns:
        the affine matrix in the normalized style.
    Notes:
        Inverse of `unnormalize_affine`.
    """
    # lta is affine matrix = (ndim = 2 for 2D and 3 for 3D (3x3 and 4x4 shape))
    # affine = matrix; ndim = 2 for 2D image, 3 for 3D image
    source_format, ndim, affine_shape = "(n+1)x(n+1)", affine.shape[-1] - 1, affine.shape[:-2]
    _affine = convert_affine(affine, "(n+1)x(n+1)", source_format)

    _target_shape = source_shape if target_shape is None else target_shape

    if ndim == 2:
        # here, the z-axis definition is changed between ndimage and torch, so positive rotation is not the same.
        _affine = _affine[..., [1, 0, 2]]
        _affine = _affine[..., [1, 0, 2], :]
    else:
        print(f"ndim = {ndim} is not tested, might have different conventions")

    # the input affine is about the corner, and in image coordinates (with corner voxels aligned in the center of the
    # voxel, i.e. ndimage assumes align_corners == True).
    # the output affine should be about the center, in relative coordinates (with corner voxels aligned along the edge).

    # first: rescale from relative coordinates (-1, 1) --> (-0.5, (n-1)+0.5)
    scale_factor_source = np.asarray(source_shape, dtype=_affine.dtype) / 2.
    scale_factor_target = np.asarray(_target_shape, dtype=_affine.dtype) / 2.
    source2relative = convert_affine(scale_factor_target, "(n+1)x(n+1)", "n")
    # move center of source2relative
    source2relative[..., :-1, -1] += scale_factor_target - 0.5
    # second: apply affine
    _affine = np.matmul(_affine, source2relative)
    # third: rescale to relative coordinates (-0.5, (n-1)+0.5) --> (-1, 1)
    relative2target = convert_affine(1. / scale_factor_source, "(n+1)x(n+1)", "n")
    relative2target[..., :-1, -1] += -1 + 0.5 / scale_factor_source
    # move center of relative2target
    _affine = np.matmul(relative2target, _affine)

    return convert_affine(_affine, affine_format, "(n+1)x(n+1)")


def affine_pytorch_style(affine_info, plane, source_shape, target_shape, normalized=False):
    # create affine from rot and tl infos
    rot, tlx, tly = affine_info
    affine_rot = np.asarray([[math.cos(rot), -math.sin(rot), 0],
                             [math.sin(rot), math.cos(rot), 0],
                             [0, 0, 1]])

    affine_tl = np.asarray([[1, 0, -tlx],
                            [0, 1, -tly],
                            [0, 0, 1]])

    # Reassemble to match affine style (first rotation, then translation; reverse in FS)
    if plane == "axial":
        affine = affine_rot @ affine_tl
    else:
        affine = affine_rot.T @ affine_tl

    # Normalize and convert affine to correct pytorch format
    if not normalized:
        affine = normalize_affine(affine, source_shape=source_shape, target_shape=target_shape,
                                  affine_format="(n+1)x(n+1)")
        affine = convert_affine(affine, 'nx(n+1)', "(n+1)x(n+1)")

    return affine
