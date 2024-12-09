# VINNA4neonates Flags
Next, you will need to select the `*vinna4neonates-flags*` and replace `*vinna4neonates-flags*` with your options. Please see the Examples below for some example flags.

The `*vinna4neonates-flags*` will usually include the subject directory (`--sd`; Note, this will be the mounted path - `/output` - for containers), the subject name/id (`--sid`) and the path to the input image (`--t1` or `--t2). For example:

```bash
... --sd /output --sid test_subject --t1 /data/test_subject_t1.nii.gz
```

Alternatively, you can change --t1 to --t2 if you have a T2-weighted image:
```bash
... --sd /output --sid test_subject --t2 /data/test_subject_t2.nii.gz
```
In the following, we give an overview of the most important options, but you can view a full list of options with 

```bash
./run_vinna4neonates.sh --help
```


## Required arguments
* `--sd`: Output directory \$SUBJECTS_DIR (equivalent to FreeSurfer setup --> $SUBJECTS_DIR/sid/mri will be created).
* `--sid`: Subject ID for directory inside \$SUBJECTS_DIR to be created ($SUBJECTS_DIR/sid/...)
* Either `--t1`: T1 full head input (not bias corrected, global path). 
* Or `--t2`: T2 full head input (not bias corrected, global path). 

The network was trained with conformed images (UCHAR, 256x256x256, 1-0.5 mm voxels and standard slice orientation). 
These specifications are checked in the run_prediction.py script and the image is automatically conformed if it does 
not comply. Note, outputs will be in the conformed space (as in FastSurfer). 

If you image resolution results in a size above 320, you must adjust the --out_dims flag (set to max size of your image).

## Segmentation arguments (optional)
* `--seg_log`: Name and location for the log-file for the segmentation (VINNA4neonates). Default: $SUBJECTS_DIR/$sid/scripts/deep-seg.log
* `--viewagg_device`: Define where the view aggregation should be run on. Can be "auto" or a device (see --device). By default, the program checks if you have enough memory to run the view aggregation on the gpu. The total memory is considered for this decision. If this fails, or you actively overwrote the check with setting with "cpu" view agg is run on the cpu. Equivalently, if you pass a different device, view agg will be run on that device (no memory check will be done).
* `--device`: Select device for NN segmentation (_auto_, _cpu_, _cuda_, _cuda:<device_num>_, _mps_), where cuda means Nvidia GPU, you can select which one e.g. "cuda:1". Default: "auto", check GPU and then CPU. "mps" is for native MAC installs to use the Apple silicon (M-chip) GPU. 
* `--segfile`: Name of the segmentation file, which includes the aparc+DKTatlas-aseg segmentations. Requires an ABSOLUTE Path! Default location: \$SUBJECTS_DIR/\$sid/mri/aparc.DKTatlas+aseg.deep.mgz
* `--out_dims`: Change if the image is too large for set padding size (i.e. if resolution smaller 0.5 --> 520). Default: 320.

## Other
* `--threads`: Target number of threads for segmentation, `1` (default) forces VINNA4neonates to only really use one core. Note, that the default value may change in the future for better performance on multi-core architectures.
* `--vox_size`: Forces processing at a specific voxel size. If a number between 0.7 and 1 is specified (below is experimental) the image is conformed to that isotropic voxel size and processed. 
  If "min" is specified (default), the voxel size is read from the size of the minimal voxel size (smallest per-direction voxel size) in the image:
  If the minimal voxel size is bigger than 0.98mm, the image is conformed to 1mm isometric.
  If the minimal voxel size is smaller or equal to 0.98mm, the image will be conformed to isometric voxels of that voxel size.
* `--py`: Command for python, used in both pipelines. Default: python3
* `--batch`: Batch size for inference. Default: 1
* `--conformed_name`: Name of the file in which the conformed input image will be saved. Default location: \$SUBJECTS_DIR/\$sid/mri/orig.mgz
* `-h`, `--help`: Prints help text
