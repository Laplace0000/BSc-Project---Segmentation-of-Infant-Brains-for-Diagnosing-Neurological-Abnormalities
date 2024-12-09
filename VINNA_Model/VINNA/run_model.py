from os.path import join
import sys
import argparse
import json
import h5py

import VINNA.utils.misc as misc
from VINNA.utils.load_config import get_config
from VINNA.train import Trainer


def setup_options():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')

    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="../FastInfantSurfer/configs/FastSurferVINN_net1.yaml",
        type=str,
    )
    parser.add_argument("--aug", action='append', help="List of augmentations to use.", default=None)

    parser.add_argument("--opt", action='append', default=["aseg", "children"],
                        help="Specify types of classes (adults/children, aseg/aparc.")

    parser.add_argument(
        "opts",
        help="See VINNA/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def main():
    args = setup_options()
    cfg = get_config(args)

    if args.aug is not None:
        cfg.DATA.AUG = args.aug
    cfg.DATA.CLASS_OPTIONS = args.opt

    summary_path = misc.check_path(join(cfg.LOG_DIR, 'summary'))
    if cfg.EXPR_NUM == "Default":
        cfg.EXPR_NUM = str(misc.find_latest_experiment(join(cfg.LOG_DIR, 'summary')) + 1)

    if cfg.TRAIN.RESUME and cfg.TRAIN.RESUME_EXPR_NUM != "Default":
        cfg.EXPR_NUM = cfg.TRAIN.RESUME_EXPR_NUM

    cfg.SUMMARY_PATH = misc.check_path(join(summary_path, cfg.MODEL.MODEL_NAME, cfg.DATA.AUGNAME, '{}'.format(cfg.EXPR_NUM)))
    cfg.CONFIG_LOG_PATH = misc.check_path(join(cfg.LOG_DIR, "config", cfg.MODEL.MODEL_NAME, cfg.DATA.AUGNAME, '{}'.format(cfg.EXPR_NUM)))

    with h5py.File(cfg.DATA.PATH_HDF5_TRAIN, "r") as hf:
        cfg.DATA.SIZES = [str(a) for a in hf.keys()]

    with open(join(cfg.CONFIG_LOG_PATH, "config.yaml"), "w") as json_file:
        json.dump(cfg, json_file, indent=2)

    trainer = Trainer(cfg=cfg)
    trainer.run()


if __name__ == '__main__':
    main()
    # print(pprint.pformat(cfg))