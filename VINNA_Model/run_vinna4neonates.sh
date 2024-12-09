#!/bin/bash

# Copyright 2023 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

VERSION='$Id$'

# Set default values for arguments
if [[ -z "${BASH_SOURCE[0]}" ]]; then
    THIS_SCRIPT="$0"
else
    THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
if [[ -z "$VINNA4NEONATES_HOME" ]]
then
  VINNA4NEONATES_HOME=$(cd "$(dirname "$THIS_SCRIPT")" &> /dev/null && pwd)
  echo "Setting ENV variable VINNA4NEONATES_HOME to script directory ${VINNA4NEONATES_HOME}. "
  echo "Change via environment to location of your choice if this is undesired (export VINNA4NEONATES_HOME=/dir/to/VINNA4NEONATES)"
  export VINNA4NEONATES_HOME
fi

vinnadir="$VINNA4NEONATES_HOME/VINNA"

# Regular flags defaults
subject=""
t1=""
t2=""
segfile=""
seg_default="\$SUBJECTS_DIR/\$SID/mri/vinna.mgz"
statsfile=""
conformed_name=""
seg_log=""
viewagg="auto"
device="auto"
batch_size="1"
out_dims=320
vox_size="min"
threads="1"
# python3.10 -s excludes user-directory package inclusion
python="python3 -s"
allow_root=()

function usage()
{
cat << EOF

Usage: run_vinna4neonates.sh --sid <sid> --sd <sdir> --t1 <t1_input> or --t2 <t2_input> [OPTIONS]

run_vinna4neonates.sh takes a T1 or T2 full head image and creates a segmentation
using VINNA (equivalent to dHCP \${sid}_\${ses}_drawem_all_labels.nii.gz)

FLAGS:

  --sid <subjectID>       Subject ID to create directory inside \$SUBJECTS_DIR
  --sd  <subjects_dir>    Output directory \$SUBJECTS_DIR (or pass via env var)
  --t1  <T1_input>        T1 full head input (not bias corrected). Requires an
                            ABSOLUTE Path!
  --t2  <T2_input>        T2 full head input (not bias corrected). Requires an
                            ABSOLUTE Path!
  --id <input directory>  Data input directory where orig input image is located.
  --segfile <filename>    Name of the segmentation file, which includes the
                            87-label segmentations.
                            Requires an ABSOLUTE Path! Default location:
                            $seg_default
  --vox_size <0.5-1|min>  Forces processing at a specific voxel size.
                            If a number between 0.5 and 1 is specified (below
                            is experimental) the T1w image is conformed to
                            that voxel size and processed.
                            If "min" is specified (default), the voxel size is
                            read from the size of the minimal voxel size
                            (smallest per-direction voxel size) in the T1w
                            image:
                              If the minimal voxel size is bigger than 0.98mm,
                                the image is conformed to 1mm isometric.
                              If the minimal voxel size is smaller or equal to
                                0.98mm, the T1w image will be conformed to
                                isometric voxels of that voxel size.
  --out_dims              Change if the image is too large for set padding size
                            (e.g. for resolution 0.5, this should be 520).
                            Default: 320.
  --seg_log <seg_log>     Log-file for the segmentation (VINNA4neonates)
                            Default: \$SUBJECTS_DIR/\$SID/scripts/vinna-seg.log

  --conformed_name <conf.mgz>
                          Name of the file in which the conformed input
                            image will be saved. Requires an ABSOLUTE Path!
                            Default location:
                            \$SUBJECTS_DIR/\$SID/mri/orig.mgz.
  -h --help               Print Help

Resource Options:
  --device <str>          Set device on which inference should be run ("cpu" for
                            CPU, "cuda" for Nvidia GPU, or pass specific device,
                            e.g. "cuda:1"), default check whether a GPU is
                            available, then fallback to CPU.
  --viewagg_device <str>  Define where the view aggregation should be run on.
                            Can be "auto" or a device (see --device). By default,
                            the program checks if you have enough memory to run
                            the view aggregation on the gpu. The total memory is
                            considered for this decision. If this fails, or you
                            actively overwrote the check with setting with "cpu"
                            view agg is run on the cpu. Equivalently, if you
                            pass a different device, view agg will be run on that
                            device (no memory check will be done).
                            ATTENTION: "auto" only works reliably for "out_dims"
                                       up to 320.
  --batch <batch_size>    Batch size for inference. Default: 1
  --py <python_cmd>       Command for python, used in both pipelines.
                            Default: "$python"
                            (-s: do no search for packages in home directory)


REFERENCES:

If you use this for research publications, please cite:

Henschel L, Kuegler D, Zoellei L*, Reuter M* (*co-last),
 Orientation Independence through Latent Augmentation,
 Imaging Neuroscience, (2023), arxiv.

Henschel L, Conjeti S, Estrada S, Diers K, Fischl B, Reuter M, FastSurfer - A
 fast and accurate deep learning based neuroimaging pipeline, NeuroImage 219
 (2020), 117012. https://doi.org/10.1016/j.neuroimage.2020.117012

Henschel L*, Kuegler D*, Reuter M. (*co-first). FastSurferVINN: Building
 Resolution-Independence into Deep Learning Segmentation Methods - A Solution
 for HighRes Brain MRI. NeuroImage 251 (2022), 118933.
 http://dx.doi.org/10.1016/j.neuroimage.2022.118933

EOF
}

# PRINT USAGE if called without params
if [[ $# -eq 0 ]]
then
  usage
  exit
fi

# PARSE Command line
inputargs=("$@")
POSITIONAL=()
while [[ $# -gt 0 ]]
do
# make key lowercase
key=$(echo "$1" | tr '[:upper:]' '[:lower:]')

case $key in
    --sid)
    subject="$2"
    shift # past argument
    shift # past value
    ;;
    --sd)
    sd="$2"
    shift # past argument
    shift # past value
    ;;
    --t1)
    tw="$2"
    mode="T1"
    shift # past argument
    shift # past value
    ;;
    --t2)
    tw="$2"
    mode="T2"
    shift # past argument
    shift # past value
    ;;
    --segfile)
    segfile="$2"
    shift # past argument
    shift # past value
    ;;
    --conformed_name)
    conformed_name="$2"
    shift # past argument
    shift # past value
    ;;
    --seg_log)
    seg_log="$2"
    shift # past argument
    shift # past value
    ;;
    --viewagg_device)
    case "$2" in
      check)
      echo "WARNING: the option \"check\" is deprecated for --viewagg_device <device>, use \"auto\"."
      viewagg="auto"
      ;;
      gpu)
      viewagg="cuda"
      ;;
      *)
      viewagg="$2"
      ;;
    esac
    shift # past argument
    shift # past value
    ;;
    --device)
    device=$2
    shift # past argument
    shift # past value
    ;;
    --batch)
    batch_size="$2"
    shift # past argument
    shift # past value
    ;;
    --vox_size)
    vox_size="$2"
    shift # past argument
    shift # past value
    ;;
    --out_dims)
    out_dims="$2"
    shift # past argument
    shift # past value
    ;;
    --py)
    python="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    usage
    exit
    ;;
    *)    # unknown option
    if [[ "$1" == "" ]]
    then
      # skip empty arguments
      shift
    else
      echo "ERROR: Flag '$1' unrecognized."
      exit 1
    fi
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# make sure VINNA4neonates is in the PYTHONPATH
if [[ "$PYTHONPATH" == "" ]]
then
  export PYTHONPATH="$VINNA4NEONATES_HOME"
else
  export PYTHONPATH="$VINNA4NEONATES_HOME:$PYTHONPATH"
fi

# make sure the python  executable is valid and found
if [[ -z "$(which "${python/ */}")" ]]; then
    echo "Cannot find the python interpreter ${python/ */}."
    exit 1
fi

# Warning if run as root user
if [[ "${#allow_root}" == 0 ]] && [[ "$(id -u)" == "0" ]]
  then
    echo "You are trying to run '$0' as root. We advice to avoid running VINNA4neonates as root, "
    echo "because it will lead to files and folders created as root."
    echo "If you are running FastSurfer in a docker container, you can specify the user with "
    echo "'-u \$(id -u):\$(id -g)' (see https://docs.docker.com/engine/reference/run/#user)."
    echo "If you want to force running as root, you may pass --allow_root to run_fastsurfer.sh."
    exit 1;
fi

# CHECKS
if { [[ -z "$tw" ]] || [[ ! -f "$tw" ]]; }
  then
    echo "ERROR: Neither T1 image ($t1) nor T2 image ($t2) could be found."
    echo "Must supply an existing T1 or T2 input (full head) via "
    echo "--t1 or --t2 (absolute path and name) for generating the segmentation."
    echo "NOTES: If running in a container, make sure symlinks are valid!"
    exit 1;
fi

if [[ -z "$subject" ]]
  then
    echo "ERROR: must supply subject name via --sid"
    exit 1;
fi

if [[ -z "$segfile" ]]
  then
    segfile="${seg_default//\$SUBJECTS_DIR/${sd}}"
    segfile="${segfile//\$SID/${subject}}"
fi

if [[ -z "$conformed_name" ]]
  then
    conformed_name="${sd}/${subject}/mri/orig.mgz"
fi

if [[ -z "$seg_log" ]]
 then
    seg_log="${sd}/${subject}/scripts/deep-seg.log"
fi

if [[ -z "$build_log" ]]
 then
    build_log="${sd}/${subject}/scripts/build.log"
fi

if [[ -z "$PYTHONUNBUFFERED" ]]
then
  export PYTHONUNBUFFERED=0
fi

# check the vox_size setting
if [[ "$vox_size" =~ ^[0-9]+([.][0-9]+)?$ ]]
then
  # a number
  if (( $(echo "$vox_size < 0" | bc -l) || $(echo "$vox_size > 1" | bc -l) ))
  then
    echo "ERROR: negative voxel sizes and voxel sizes beyond 1 are not supported."
    exit 1;
  elif (( $(echo "$vox_size < 0.5" | bc -l) ))
  then
    echo "WARNING: support for voxel sizes smaller than 0.5mm iso. is experimental."
  fi
elif [[ "$vox_size" != "min" ]]
then
  # not a number or "min"
  echo "Invalid option for --vox_size, only a number or 'min' are valid."
  exit 1;
fi

if [[ "${segfile: -3}" != "${conformed_name: -3}" ]]
  then
    echo "ERROR: Specified segmentation output and conformed image output do not have same file type."
    echo "You passed --segfile ${asegdkt_segfile} and --conformed_name ${conformed_name}."
    echo "Make sure these have the same file-format and adjust the names passed to the flags accordingly!"
    exit 1;
fi

########################################## START ########################################################
mkdir -p "$(dirname "$seg_log")"

if [[ -f "$seg_log" ]]; then log_existed="true"
else log_existed="false"
fi

### IF THE SCRIPT GETS TERMINATED, ADD A MESSAGE
trap "{ echo \"run_vinna4neonates.sh terminated via signal at \$(date -R)!\" >> \"$seg_log\" ; }" SIGINT SIGTERM

# create the build log, file with all version info in parallel
printf "%s %s\n%s\n" "$THIS_SCRIPT" "${inputargs[*]}" "$(date -R)" >> "$build_log"


# "============= Running VINNA4neonates (Creating Segmentation vinna.mgz) ==============="
# use VINNA to create cortical + subcortical segmentation into 89 classes.
echo "Log file for segmentation VINNA/run_prediction.py" >> "$seg_log"
date  |& tee -a "$seg_log"
echo "" |& tee -a "$seg_log"

cmd=($python "$vinnadir/run_prediction.py" --tw "$tw" --mode "$mode"
      --segfile "$segfile" --conformed_name "$conformed_name" --out_dims "$out_dims"
      --sid "$subject" --seg_log "$seg_log" --vox_size "$vox_size"
      --batch_size "$batch_size" --viewagg_device "$viewagg" --device "$device" "${allow_root[@]}")
# specify the subject dir $sd, if asegdkt_segfile explicitly starts with it
if [[ "$sd" == "${segfile:0:${#sd}}" ]]; then cmd=("${cmd[@]}" --sd "$sd"); fi

echo "${cmd[@]}" |& tee -a "$seg_log"
"${cmd[@]}"
exit_code="${PIPESTATUS[0]}"
if [[ "${exit_code}" -ne 0 ]]
    then
      echo "ERROR: VINNA4neonates segmentation failed."
      exit 1
fi

########################################## End ########################################################
