#!/usr/bin/env python3

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

import argparse
from functools import lru_cache
from typing import Optional, List, Union

from VINNA.utils import PLANES
from VINNA.utils.checkpoint import (
    check_and_download_ckpts,
    get_checkpoints,
    load_checkpoint_config_defaults,
    YAML_DEFAULT as VINNA_YAML,
    )


class ConfigCache:
    @lru_cache
    def vinna_url(self):
        return load_checkpoint_config_defaults("url", filename=VINNA_YAML)

defaults = ConfigCache()


def make_arguments():
    parser = argparse.ArgumentParser(
        description="Check and Download Network Checkpoints"
    )
    parser.add_argument(
        "--vinna",
        default=False,
        action="store_true",
        help="Check and download VINNA default checkpoints",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=f"Specify you own base URL. This is applied to all models. \n"
             f"Default for VINNA: {defaults.vinna_url()}",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Checkpoint file paths to download, e.g. "
             "checkpoints/vinna4neonates_axial_t1_v1.0.0.pkl ...",
    )
    return parser.parse_args()


def main(
        vinna: bool,
        files: List[str],
        url: Optional[str] = None,
) -> Union[int, str]:
    if not vinna and not files:
        return ("Specify either files to download or --vinna "
                "see help -h.")

    try:
        # VINNA4neonates checkpoints
        if vinna:
            for ckpt in ["checkpoint_t1", "checkpoint_t2"]:
                vinna_config = load_checkpoint_config_defaults(
                    ckpt,
                    filename=VINNA_YAML,
                )
                get_checkpoints(
                    *(vinna_config[plane] for plane in PLANES),
                    urls=defaults.vinna_url() if url is None else [url]
                )
        for fname in files:
            check_and_download_ckpts(
                fname,
                urls=defaults.all_urls() if url is None else [url],
            )
    except Exception as e:
        from traceback import print_exception
        print_exception(e)
        return e.args[0]
    return 0


if __name__ == "__main__":
    import sys
    from logging import basicConfig, INFO

    basicConfig(stream=sys.stdout, level=INFO)
    args = make_arguments()
    sys.exit(main(**vars(args)))