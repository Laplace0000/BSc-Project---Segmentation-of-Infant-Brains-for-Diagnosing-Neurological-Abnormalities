import time
import glob
from os.path import join, basename
from collections import defaultdict
import numpy as np
import nibabel as nib
import h5py
import pandas as pd

import sys
import FastSurfer.FastSurferCNN.data_loader.conform as conf
import FastSurfer.FastSurferCNN.data_loader.data_utils as du

from VINNA.data_processing.utils.data_utils import transform_axial, transform_sagittal, \
    create_weight_mask, read_classes_from_lut, \
    get_thick_slices, filter_blank_slices_thick, bounding_box_slices, bounding_box_crop, \
    read_lta, rotation_matrix_to_euler

LookUpRes = {0.5: "_min", 0.8: "_08_dhcp", 1.0: "", 0.7: ""}
ROT_TL_INDICES = {"coronal": [-1, 0, 1],
                  "axial": [1, 0, 1],
                  "sagittal": [0, 1, 2]}

class H5pyDataset:

    def __init__(self, params):

        self.dataset_name = params["dataset_name"]
        self.data_path = params["data_path"]
        self.slice_thickness = params["thickness"]
        self.orig_name = params["image_name"].rsplit(".")[0]
        self.orig_ftype = params["image_name"][len(self.orig_name):]
        self.lta_name = "t2_upright"
        self.t2_name = params["t2_name"].rsplit(".")[0]
        self.t2_ftype = params["t2_name"][len(self.t2_name):]
        self.aparc_name = params["gt_name"].rsplit(".")[0]
        self.aparc_ftype = params["gt_name"][len(self.aparc_name):]
        self.aparc_nocc = params["gt_nocc"].rsplit(".")[0] if params["gt_nocc"] is not None else None
        self.aparc_nocc_ftype = params["gt_nocc"][len(self.aparc_nocc):] if params["gt_nocc"] is not None else None
        self.mapping = params["mapping"]

        self.lut = read_classes_from_lut(params["lut"])
        self.labels, self.labels_sag = du.get_labels_from_lut(self.lut, params["sag-mask"])
        self.lateralization = du.unify_lateralized_labels(self.lut, params["combi"])
        gm_labels_mask = self.lut["LabelName"].str.endswith("_GM")
        self.ctx_full = self.lut["MappedID"][gm_labels_mask].values
        self.ctx_sag = self.lut["SagID"][gm_labels_mask].values
        external_labels_mask = self.lut["LabelName"].isin(["Extra_cranial_background", "CSF"])
        self.external_labels_full = self.lut["MappedID"][external_labels_mask].values
        self.external_labels_sag = self.lut["SagID"][external_labels_mask].values

        self.available_sizes = params["sizes"]
        self.scale_only = params["scale_only"]
        self.hires = params["hires"]
        self.add_sbj = params["add_sbj"]
        self.crop_size = params["crop_size"]
        self.img_dim = params["img_dim"]
        self.max_weight = params["max_weight"]
        self.edge_weight = params["edge_weight"]
        self.hires_weight = params["hires_weight"]
        self.smooth_weight = params["smooth_weights"]
        self.gradient = params["gradient"]
        self.gm_mask = params["gm_mask"]

        if params["csv_file"] is not None:
            separator = "\t" if params["csv_file"][-3] == "t" else ","
            s_file = pd.read_csv(params["csv_file"], sep=separator)
            if self.data_path is not None:
                s_file["Path"] = self.data_path + s_file["SubjectFix"]
            self.subject_dirs = s_file["Path"].to_list()
            # If fixed res is 0.0 = use resolution from csv, otherwise set to fixed resolution
            self.res = s_file["Resolution"].to_list() if params["fixed_res"] == 0.0 \
                else [params["fixed_res"] for _ in range(len(self.subject_dirs))]

        else:
            self.search_pattern = join(self.data_path, params["pattern"])
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)

    def check_and_conform_img(self, sbj_dir, name):
        img_d = nib.load(join(sbj_dir, name))
        if not conf.is_conform(img_d, conform_vox_size=self.hires, check_dtype=True):
            if self.scale_only:
                img = np.asanyarray(img_d.dataobj)
                src_min, scale = conf.getscale(img, 0, 255)
                img = conf.scalecrop(img, 0, 255, src_min, scale)
                img = np.uint8(np.rint(img))
            else:
                img_d = conf.conform(img_d, conform_vox_size=self.hires)
                img = np.asarray(img_d.get_fdata(), dtype=np.uint8)
        else:
            img = np.asarray(img_d.get_fdata(), dtype=np.uint8)
        return img, img_d.header

    def _load_volumes(self, subject_path, plane, index):
        # Load the orig and extract voxel spacing information (x, y, and z dim)
        res = self.res[index]
        res_suffix = LookUpRes[res]
        print('Processing ground truth segmentation {} at res {}'.format(self.aparc_name, res_suffix))

        # join names with resolution and load files
        oname = self.orig_name + res_suffix + self.orig_ftype
        orig, header = self.check_and_conform_img(subject_path, oname)
        t2name = self.t2_name + res_suffix + self.t2_ftype
        t2, _ = self.check_and_conform_img(subject_path, t2name)
        zoom = header.get_zooms()
        # Comparison between np.float32 (=type in zoom) and tuple with type float evaluates to false
        # Probably because the np.float32 is stored differently. zoom[0].item() results in approximate float
        # (e.g. 0.699999998 for a value of 0.7)
        if zoom != (np.float32(res), ) * 3:
            print(f"ERROR: Resolutions do not match!! {zoom} with dtype {type(zoom[0])} versus {(res, res, res)} with dtype {type(res)}.")
            zoom = (res, res, res)

        # Load the segmentation ground truth - dhcp suffix only exists for T1w and T2w image, not GT for 0.8
        if self.add_sbj:
            aname = join(subject_path, basename(subject_path) + self.aparc_name + res_suffix.rstrip("_dhcp") + self.aparc_ftype)
        else:
            aname = join(subject_path, self.aparc_name + res_suffix.rstrip("_dhcp") + self.aparc_ftype)

        aseg = np.asanyarray(nib.load(aname).dataobj, dtype=np.int16)
        images_dict = {"T1": orig, "T2": t2, "Labels": aseg}

        if self.aparc_nocc is not None:
            aseg_nocc = nib.load(join(subject_path, self.aparc_nocc + res_suffix + self.aparc_nocc_ftype))
            aseg_nocc = np.asarray(aseg_nocc.get_fdata(), dtype=np.int16)
            images_dict["aseg_no_cc"] = aseg_nocc
        else:
            aseg[((aseg >= 251) & (aseg <= 255))] = 2

        if self.crop_size is not None and orig.shape[0] != self.crop_size:
            print("Cropping images from {0} to {1}x{1}x{1}".format(orig.shape, self.crop_size))
            # Slice images to unified size (e.g. 256x256x256)
            bb_box_slices_start, bb_box_slices_stop = bounding_box_slices(aseg, size_half=self.crop_size // 2)
            images_dict = bounding_box_crop(images_dict, bb_box_slices_start, bb_box_slices_stop)
            print("finished cropping")

        if plane == 'sagittal':
            zoom = zoom[::-1][:2]

        elif plane == 'coronal':
            zoom = zoom[:2]

        elif plane == 'all':
            zoom = zoom
        else:
            zoom = zoom[1:]
        return images_dict, zoom

    def transform(self, plane, imgs):

        for name, img in imgs.items():
            if plane == "sagittal":
                imgs[name] = transform_sagittal(img)
            elif plane == "axial":
                imgs[name] = transform_axial(img)
            else:
                pass
        return imgs

    def prepare_slices(self, plane, img):
        # transform volumes to correct shape
        img = self.transform(plane, img)

        # Create Thick Slices, filter out blanks
        img["T1"] = get_thick_slices(img["T1"], self.slice_thickness)
        if self.t2_name is not None:
            img["T2"] = get_thick_slices(img["T2"], slice_thickness=self.slice_thickness)

        # After this step, label vol is the last image
        imgs = filter_blank_slices_thick(img, threshold=2)

        for name, img in imgs.items():
            if name == "T1" or (name == "T2" and self.t2_name is not None):
                imgs[name] = np.transpose(img, (2, 0, 1, 3))  # D=B x H x W x C
            else:
                imgs[name] = np.transpose(img, (2, 0, 1))  # D=B x H x W

        return imgs, imgs["T1"].shape[1]

    def prepare_volume(self, imgs):
        # add dim at position 0 (BxHxWxDxC)
        for name, img in imgs.items():
            if name == "T1" or (name == "T2" and self.t2_name is not None):
                imgs[name] = np.expand_dims(img, axis=[0, -1])  # B=1, H, W, D, C
            else:
                imgs[name] = np.expand_dims(img, axis=[0])  # B=1, H, W, D

        return imgs, imgs["T1"].shape[1]

    def prepare_imgs(self, plane, imgs):
        if self.img_dim == 2:
            return self.prepare_slices(plane, imgs)
        else:
            return self.prepare_volume(imgs)

    def get_affine_elements(self, subject_path, index, plane):
        res_suffix = LookUpRes[self.res[index]] + ".lta"
        lta_f = join(subject_path, self.lta_name + res_suffix)
        lta = read_lta(lta_f)
        rotation = rotation_matrix_to_euler(lta["lta_vox2vox"][:3, :3], angles=False)
        translation = lta["lta_vox2vox"][:-1, -1]

        ri, tlxi, tlyi = ROT_TL_INDICES[plane]
        rot = rotation[ri]
        tlx, tly = translation[tlxi], translation[tlyi]
        return np.asarray([rot, tlx, tly])

    def create_hdf5_dataset(self, plane='axial'):
        data_per_size = defaultdict(lambda: defaultdict(list))
        start_d = time.time()

        for idx, current_subject in enumerate(self.subject_dirs):
            try:
                start = time.time()

                print("Volume Nr: {} Processing MRI Data from {}/{}".format(idx + 1, current_subject, self.orig_name))

                img_dict, zoom = self._load_volumes(current_subject, plane, idx)

                # assert size in self.available_sizes, f"The input size {size} is not expected, available size are {self.available_sizes}"
                mapped_aseg, mapped_aseg_sag = du.map_aparc_aseg2label(img_dict["Labels"], aseg_nocc=None,
                                                                       processing=self.mapping,
                                                                       labels=self.labels, labels_sag=self.labels_sag,
                                                                       sagittal_lut_dict=self.lateralization
                                                                       )

                if plane == 'sagittal':
                    mapped_aseg = mapped_aseg_sag
                    weights = create_weight_mask(mapped_aseg, self.labels_sag, max_weight=self.max_weight,
                                                 ctx_labels=self.ctx_sag, ext=self.external_labels_sag,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 mean_filter=self.smooth_weight, use_cortex_mask=self.gm_mask,
                                                 gradient=self.gradient)

                else:
                    weights = create_weight_mask(mapped_aseg, self.labels, max_weight=self.max_weight,
                                                 ctx_labels=self.ctx_full, ext=self.external_labels_full,
                                                 max_edge_weight=self.edge_weight, max_hires_weight=self.hires_weight,
                                                 mean_filter=self.smooth_weight, use_cortex_mask=self.gm_mask,
                                                 gradient=self.gradient)
                img_dict["Labels"] = mapped_aseg
                img_dict["Weights"] = weights
                print("Created weights with max_w {}, gradient {},"
                      " edge_w {}, hires_w {}, gm_mask {}, mean filter {}".format(self.max_weight, self.gradient,
                                                                                  self.edge_weight,
                                                                                  self.hires_weight, self.gm_mask,
                                                                                  self.smooth_weight))

                volumes, size = self.prepare_imgs(plane, img_dict)
                #affine_elements = self.get_affine_elements(current_subject, idx, plane)
                data_per_size[f'{size}']['orig'].extend(volumes["T1"])
                data_per_size[f'{size}']['t2'].extend(volumes["T2"])
                data_per_size[f'{size}']['aseg'].extend(volumes["Labels"])
                data_per_size[f'{size}']['weight'].extend(volumes["Weights"])
                #data_per_size[f'{size}']['upr_affine'].extend([affine_elements for _ in range(volumes["T1"].shape[0])])
                data_per_size[f'{size}']['zoom'].extend((zoom,) * volumes["T1"].shape[0])
                sub_name = current_subject.split("/")[-1]
                data_per_size[f'{size}']['subject'].append(sub_name.encode("ascii", "ignore"))

            except FileNotFoundError as e:
                print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                continue

        for key, data_dict in data_per_size.items():
            data_per_size[key]['orig'] = np.asarray(data_dict['orig'], dtype=np.uint8)
            data_per_size[key]['t2'] = np.asarray(data_dict['t2'], dtype=np.uint8)
            data_per_size[key]['aseg'] = np.asarray(data_dict['aseg'], dtype=np.uint8)
            data_per_size[key]['weight'] = np.asarray(data_dict['weight'], dtype=np.float32)
            #data_per_size[key]['upr_affine'] = np.asarray(data_dict['upr_affine'], dtype=np.float32)

        with h5py.File(self.dataset_name, "w") as hf:
            dt = h5py.special_dtype(vlen=str)
            for key, data_dict in data_per_size.items():
                group = hf.create_group(f"{key}")
                group.create_dataset("orig_dataset", data=data_dict['orig'])  # , compression='lzf')
                group.create_dataset("t2_dataset", data=data_dict['t2'])  # , compression='lzf')
                group.create_dataset("aseg_dataset", data=data_dict['aseg'])  # , compression='lzf')
                group.create_dataset("weight_dataset", data=data_dict['weight'])  # , compression='lzf')
                #group.create_dataset("upr_affine_dataset", data=data_dict['upr_affine'])
                group.create_dataset("zoom_dataset", data=data_dict['zoom'])
                group.create_dataset("subject", data=data_dict['subject'], dtype=dt)  # , compression='lzf')

        end_d = time.time() - start_d
        print("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))


if __name__ == '__main__':
    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str,
                        default="/groups/ag-reuter/projects/VINNA4neonates/experiments/hdf5_sets/training_bigMixAffine_dHCP_coronal.hdf5",
                        help='path and name of hdf5-data_loader (default: ../data/Multires_coronal.hdf5)')
    parser.add_argument('--plane', type=str, default="coronal", choices=["axial", "coronal", "sagittal", "all"],
                        help="Which plane to put into file (axial (default), coronal, sagittal or all (for 3D))")
    parser.add_argument('--data_dir', type=str, default="/groups/ag-reuter/projects/datasets/dHCP/Data/", help="Directory with images to load")
    parser.add_argument('--thickness', type=int, default=3, help="Number of pre- and succeeding slices (default: 3)")
    parser.add_argument('--csv_file', type=str,
                        default="/groups/ag-reuter/projects/VINNA4neonates/Dataset_splits/dataset_split_large_training_t1t2_meta.tsv",
                        help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, help="Pattern to match files in directory.")
    parser.add_argument('--image_name', type=str, default="T1w.nii.gz",
                        help="Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)")
    parser.add_argument('--t2_name', type=str, default="T2w.nii.gz",
                        help="Default name of original images. FreeSurfer orig.mgz is default (mri/orig.mgz)")
    parser.add_argument('--gt_name', type=str, default="_mapped26_dseg.nii.gz",
                        help="Default name for ground truth segmentations. Default: mri/aparc.DKTatlas+aseg.mgz."
                             " If Corpus Callosum segmentation is already removed, do not set gt_nocc."
                             " (e.g. for our internal training set mri/aparc.DKTatlas+aseg.filled.mgz exists already"
                             " and should be used here instead of mri/aparc.DKTatlas+aseg.mgz). ")
    parser.add_argument('--gt_nocc', type=str, default=None,
                        help="Segmentation without corpus callosum (used to mask this segmentation in ground truth)."
                             " If the used segmentation was already processed, do not set this argument."
                             " For a normal FreeSurfer input, use mri/aseg.auto_noCCseg.mgz.")
    parser.add_argument('--crop_size', type=int, default=256,
                        help="Unified size images should be cropped to (e.g. 256). Default=None (=no cropping)")
    parser.add_argument('--max_w', type=int, default=5,
                        help="Overall max weight for any voxel in weight mask. Default=5")
    parser.add_argument('--edge_w', type=int, default=5, help="Weight for edges in weight mask. Default=5")
    parser.add_argument('--hires_w', type=int, default=None,
                        help="Weight for hires elements (sulci, WM strands, cortex border) in weight mask. Default=None")
    parser.add_argument('--smooth', action='store_true', default=False,
                        help="Turn on to smooth final weight mask with mean filter")
    parser.add_argument('--no_grad', action='store_true', default=False,
                        help="Turn on to only use median weight frequency (no gradient)")
    parser.add_argument('--gm', action="store_true", default=False,
                        help="Turn on to add cortex mask for hires-processing.")
    parser.add_argument('--img_dim', type=int, default=2, choices=[2, 3],
                        help="Image dimension for loader (either 2 for 2D net or 3 for 3D net).")
    parser.add_argument('--processing', type=str, default="none", choices=["aparc", "aseg", "none"],
                        help="Use aseg, aparc or no specific mapping processing")
    parser.add_argument('--lut', type=str, help='Path and name of lut file (FreeSurfer-like)',
                        default='/groups/ag-reuter/projects/VINNA4neonates/experiments/LUTs/FastInfantSurfer_dHCP_full_LUT.tsv')
    parser.add_argument('--combi', action='append', default=["Left-", "Right-"],
                        help="Suffixes of labels names to combine. Default: Left- and Right-.")
    parser.add_argument('--sag_mask', default=("Left-", "ctx-rh"),
                        help="Suffixes of labels names to mask for final sagittal labels. Default: Left- and ctx-rh.")
    parser.add_argument('--sizes', nargs='+', type=int, default=256)
    parser.add_argument('--scale_only', action="store_true", default=False, help="Only conform scale, not img dimensions")
    parser.add_argument('--hires', default="min",
                        help="Conform image to min size (=hires). Can specify voxelsize, if desired")
    parser.add_argument('--fix_res', default=0.0, type=float,
                        help="Read resolution for image from file (default=0.0), or set via this flag for all images (0.5, 0.8 or 1.0 e.g.)")
    parser.add_argument('--add_sbj', action="store_true", default=True,
                        help="Add subject to gt_name (name passed under --gt_name is suffix not whole name.)")

    args = parser.parse_args()

    dataset_params = {"dataset_name": args.hdf5_name, "data_path": args.data_dir, "thickness": args.thickness,
                      "csv_file": args.csv_file, "pattern": args.pattern, "image_name": args.image_name,
                      "gt_name": args.gt_name, "gt_nocc": args.gt_nocc, "sizes": args.sizes, "t2_name": args.t2_name,
                      "crop_size": args.crop_size, "max_weight": args.max_w, "edge_weight": args.edge_w,
                      "hires_weight": args.hires_w, "smooth_weights": args.smooth, "gm_mask": args.gm,
                      "gradient": not args.no_grad, "img_dim": args.img_dim, "scale_only": args.scale_only,
                      "mapping": args.processing, "lut": args.lut, "combi": args.combi, "sag-mask": args.sag_mask,
                      "add_sbj": args.add_sbj, "hires": args.hires, "fixed_res": args.fix_res
                      }
    dataset_generator = H5pyDataset(params=dataset_params)
    dataset_generator.create_hdf5_dataset(plane=args.plane)
