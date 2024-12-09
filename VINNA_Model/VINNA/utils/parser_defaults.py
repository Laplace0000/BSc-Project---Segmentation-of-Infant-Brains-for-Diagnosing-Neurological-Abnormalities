# Copyright 2022 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

"""
Contains the ALL_FLAGS dictionary, which can be used as follows to add default flags.

>>> parser = argparse.ArgumentParser()
>>> ALL_FLAGS["allow_root"](parser, dest="root")
>>> args = parser.parse_args()
>>> allows_root = args.root  # instead of the default dest args.allow_root

Values can also be extracted by
>>> print(ALL_FLAGS["allow_root"](dict, dest="root")
>>> # {'flag': '--allow_root', 'flags': ('--allow_root',), 'action': 'store_true',
>>> #  'dest': 'root', 'help': 'Allow execution as root user.'}
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Protocol, Type, TypeVar, Union

from VINNA.utils import Plane, PLANES
from FastSurferCNN.utils.arg_types import float_gt_zero_and_le_one as __conform_to_one
from FastSurferCNN.utils.arg_types import unquote_str
from FastSurferCNN.utils.arg_types import vox_size as __vox_size
from FastSurferCNN.utils.threads import get_num_threads

VINNA_ROOT = Path(__file__).parents[2]
PLANE_SHORT = {"checkpoint_t1": "ckpt", "checkpoint_t2": "ckpt", "config": "cfg"}
PLANE_HELP = {
    "checkpoint_t1": "{} checkpoint to load for inference on T1",
    "checkpoint_t2": "{} checkpoint to load for inference on T2",
    "config": "Path to the {} config file",
}
VoxSize = Union[Literal["min"], float]


class CanAddArguments(Protocol):
    """
    [MISSING].
    """

    def add_argument(self, *args, **kwargs):
        """
        [MISSING].
        """
        ...


def __arg(*default_flags, **default_kwargs):
    """
    Create stub function, which sets default settings for argparse arguments.

    The positional and keyword arguments function as if they were directly passed to
    parser.add_arguments().

    The result will be a stub function, which has as first argument a parser (or other
    object with an add_argument method) to which the argument is added. The stub
    function also accepts positional and keyword arguments, which overwrite the default
    arguments. Additionally, these specific values can be callables, which will be
    called upon the default values (to alter the default value).

    This function is private for this module.
    """

    def _stub(parser: Union[CanAddArguments, Type[Dict]], *flags, **kwargs):
        # prefer the value passed to the "new" call
        for kw, arg in kwargs.items():
            if callable(arg) and kw in default_kwargs.keys():
                kwargs[kw] = arg(default_kwargs[kw])
        # if no new value is provided to _stub (which is the callable in ALL_FLAGS), use
        # the default value (stored in the callable/passed to the default below)
        for kw, default in default_kwargs.items():
            if kw not in kwargs.keys():
                kwargs[kw] = default

        _flags = flags if len(flags) != 0 else default_flags
        if hasattr(parser, "add_argument"):
            return parser.add_argument(*_flags, **kwargs)
        elif parser == dict or isinstance(parser, dict):
            return {"flag": _flags[0], "flags": _flags, **kwargs}
        else:
            raise ValueError(
                f"Unclear parameter, should be dict or argparse.ArgumentParser, not "
                f"{type(parser).__name__}."
            )

    return _stub


ALL_FLAGS = {
    "tw": __arg(
        "--tw",
        type=str,
        dest="orig_name",
        default="mri/orig.mgz",
        help="Name of full head MRI to segment. Can be T1 or T2 (specify via --mode flag)."
             "Absolute path if single image else common image name. Default: mri/orig.mgz",
    ),
    "mode": __arg(
        "--mode",
        type=str,
        dest="mode",
        default="T2",
        help="Input image modality. Either T2 (default) or T1.",
    ),
    "remove_suffix": __arg(
        "--remove_suffix",
        type=str,
        dest="remove_suffix",
        default="",
        help="Optional: remove suffix from path definition of input file to yield "
             "correct subject name (e.g. /ses-x/anat/ for BIDS or /mri/ for FreeSurfer "
             "input). Default: do not remove anything.",
    ),
    "sid": __arg(
        "--sid",
        type=str,
        dest="sid",
        default=None,
        help="Optional: directly set the subject id to use. Can be used for single "
             "subject input. For multi-subject processing, use remove suffix if sid is "
             "not second to last element of input file passed to --tw",
    ),
    "segfile": __arg(
        "--segfile",
        type=str,
        dest="pred_name",
        default="mri/aparc.DKTatlas+aseg.deep.mgz",
        help="Name of DL-based segmentation file (similar to drawem_all_labels from dHCP). "
             "When using VINNA4neonates, this segmentation is already conformed, since "
             "inference is always based on a conformed image. Absolute path if single "
             "image else common image name. Default: mri/vinna.mgz",
    ),
    "conformed_name": __arg(
        "--conformed_name",
        type=str,
        dest="conf_name",
        default="mri/orig.mgz",
        help="Name under which the conformed input image will be saved, in the same "
             "directory as the segmentation (the input image is always conformed "
             "first, if it is not already conformed). The original input image is "
             "saved in the output directory as $id/mri/orig/001.mgz. Default: "
             "mri/orig.mgz.",
    ),
    "seg_log": __arg(
        "--seg_log",
        type=str,
        dest="log_name",
        default="",
        help="Absolute path to file in which run logs will be saved. If not set, logs "
             "will not be saved.",
    ),
    "device": __arg(
        "--device",
        default="auto",
        help="Select device to run inference on: cpu, or cuda (= Nvidia gpu) or "
             "specify a certain gpu (e.g. cuda:1), default: auto",
    ),
    "viewagg_device": __arg(
        "--viewagg_device",
        dest="viewagg_device",
        type=str,
        default="auto",
        help="Define the device, where the view aggregation should be run. By default, "
             "the program checks if you have enough memory to run the view aggregation "
             "on the gpu (cuda). The total memory is considered for this decision. If "
             "this fails, or you actively overwrote the check with setting "
             "> --viewagg_device cpu <, view agg is run on the cpu. Equivalently, if "
             "you define > --viewagg_device cuda <, view agg will be run on the gpu "
             "(no memory check will be done).",
    ),
    "in_dir": __arg(
        "--in_dir",
        type=str,
        default=None,
        help="Directory in which input volume(s) are located. Optional, if full path "
             "is defined for --tw.",
    ),
    "tag": __arg(
        "--tag",
        type=unquote_str,
        dest="search_tag",
        default="*",
        help="Search tag to process only certain subjects. If a single image should be "
             "analyzed, set the tag with its id. Default: processes all.",
    ),
    "csv_file": __arg(
        "--csv_file",
        type=str,
        help="Csv-file with subjects to analyze (alternative to --tag)",
        default=None,
    ),
    "batch_size": __arg(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference. Default=1"
    ),
    "sd": __arg(
        "--sd",
        type=str,
        default=None,
        dest="out_dir",
        help="Directory in which evaluation results should be written. Will be created "
             "if it does not exist. Optional if full path is defined for --seg_file.",
    ),
    "vox_size": __arg(
        "--vox_size",
        type=__vox_size,
        default="min",
        dest="vox_size",
        help="Choose the primary voxelsize to process, must be either a number between "
             "0 and 1 (below 0.5 is experimental) or 'min' (default). A number forces "
             "processing at that specific voxel size, 'min' determines the voxel size "
             "from the image itself (conforming to the minimum voxel size, or 1 if the "
             "minimum voxel size is above 0.95mm). ",
    ),
    "out_dims": __arg(
        "--out_dims",
        type=int,
        default=320,
        dest="out_dims",
        help="Dimensions for Padding. Need to be adjusted, if the images is larger"
             "than 320 after conforming!",
    ),
    "conform_to_1mm_threshold": __arg(
        "--conform_to_1mm_threshold",
        type=__conform_to_one,
        default=0.95,
        dest="conform_to_1mm_threshold",
        help="The voxelsize threshold, above which images will be conformed to 1mm "
             "isotropic, if the --vox_size argument is also 'min' (the --vox_size "
             "default setting). Contrary to conform.py, the default behavior of "
             "%(prog)s is to resample all images _above 0.95mm_ to 1mm.",
    ),
    "lut": __arg(
        "--lut",
        type=Path,
        help="Path and name of LUT to use.",
        default=VINNA_ROOT / "VINNA/config/VINNA4neonates_LUT.tsv",
    ),
    "allow_root": __arg(
        "--allow_root",
        action="store_true",
        dest="allow_root",
        help="Allow execution as root user.",
    ),
    "threads": __arg(
        "--threads",
        dest="threads",
        default=get_num_threads(),
        type=int,
        help=f"Number of threads to use (defaults to number of hardware threads: "
             f"{get_num_threads()})",
    ),
    "async_io": __arg(
        "--async_io",
        dest="async_io",
        action="store_true",
        help="Allow asynchronous file operations (default: off). Note, this may impact "
             "the order of messages in the log, but speed up the segmentation "
             "specifically for slow file systems.",
    ),
}

T_AddArgs = TypeVar("T_AddArgs", bound=CanAddArguments)


def add_arguments(parser: T_AddArgs, flags: Iterable[str]) -> T_AddArgs:
    """
    Add default flags to the parser from the flags list in order.

    Parameters
    ----------
    parser : T_AddArgs
        The parser to add flags to.
    flags : Iterable[str]
        The flags to add from 'device', 'viewagg_device'.

    Returns
    -------
    T_AddArgs
        The parser object.

    Raises
    ------
    RuntimeError
        If parser does not support a call to add_argument.
    """
    if not hasattr(parser, "add_argument") or not callable(parser.add_argument):
        raise RuntimeError("parser does not support add_argument!")

    for flag in flags:
        if flag.startswith("--"):
            flag = flag[2:]
        add_flag = ALL_FLAGS.get(flag, None)
        if add_flag is not None:
            add_flag(parser)
        else:
            raise ValueError(
                f"The flag '{flag}' is not defined in {add_arguments.__qualname__}."
            )
    return parser


def add_plane_flags(
    parser: argparse.ArgumentParser,
    configtype: Literal["checkpoint_t1", "checkpoint_t2", "config"],
    files: Mapping[Plane, Union[Path, str]],
    defaults_path: Union[Path, str],
) -> argparse.ArgumentParser:
    """
    Add plane arguments.

    Arguments will be added for each entry in files, where the key is the "plane"
    and the values is the file name (relative for path relative to VINNA_HOME.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to add flags to.
    configtype : Literal["checkpoint_t1", "checkpoint_t2", "config"]
        The type of files (for help text and prefix from "checkpoint" and "config".
        "checkpoint" will lead to flags like "--ckpt_{plane}_{mode}", "config" to
        "--cfg_{plane}".
    files : Mapping[Plane, Path | str]
        A dictionary of plane to filename. Relative files are assumed to be relative to
        the FastSurfer root directory.
    defaults_path : Path, str
        A path to the file to load defaults from.

    Returns
    -------
    argparse.ArgumentParser
        The parser object.
    """
    if configtype not in PLANE_SHORT:
        raise ValueError("type must be either config, checkpoint_t1, or checkpoint_t2.")

    from VINNA.utils.checkpoint import load_checkpoint_config_defaults
    defaults = load_checkpoint_config_defaults(configtype, defaults_path)

    for plane, filepath in files.items():
        path = defaults[plane] if str(filepath) == "default" else Path(filepath)
        if not path.is_absolute():
            path = VINNA_ROOT / path
        # find the first vowel in the key
        plane = plane.strip().lower()
        if plane not in PLANES:
            raise ValueError(f"Invalid key in files, no plane: {plane}")
        index = min(i for i in (plane.find(v) for v in "aeiou") if i >= 0)
        plane_short = plane[: index + 2]
        if configtype[:-3] == "checkpoint":
            # add mode modifier
            mode = configtype[-2:]
            parser.add_argument(
                f"--{PLANE_SHORT[configtype]}_{plane_short}_{mode}",
                type=Path,
                dest=f"{PLANE_SHORT[configtype]}_{plane_short}_{mode}",
                help=PLANE_HELP[configtype].format(plane),
                default=path,
            )
        else:
            parser.add_argument(
            f"--{PLANE_SHORT[configtype]}_{plane_short}",
            type=Path,
            dest=f"{PLANE_SHORT[configtype]}_{plane_short}",
            help=PLANE_HELP[configtype].format(plane),
            default=path,
            )
    return parser
