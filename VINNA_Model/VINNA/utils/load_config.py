# IMPORTS
from os.path import join, split, splitext
from VINNA.config import get_cfg_defaults


def get_config(args):
    """
    Given the arguemnts, load and initialize the configs.

    """
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.LOG_DIR = args.LOG_dir

    cfg.LOG_DIR = join(cfg.LOG_DIR)

    return cfg


def load_config(cfg_file):
    # setup base
    cfg = get_cfg_defaults()
    cfg.EXPR_NUM = "Default"
    cfg.SUMMARY_PATH = ""
    cfg.CONFIG_LOG_PATH = ""
    # Overwrite with stored arguments
    cfg.merge_from_file(cfg_file)
    return cfg
