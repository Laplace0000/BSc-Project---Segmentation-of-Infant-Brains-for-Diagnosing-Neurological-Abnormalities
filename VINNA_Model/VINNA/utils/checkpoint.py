# IMPORTS
import os
import glob
import torch
from pathlib import Path
from typing import MutableSequence, Optional, Union, Literal, TypedDict, overload, List, Dict

import requests
import yaml

from VINNA.utils import logging, Plane

Scheduler = "torch.optim.lr_scheduler"
LOGGER = logging.getLogger(__name__)

# Defaults
VINNA_ROOT = Path(__file__).parents[2]
YAML_DEFAULT = VINNA_ROOT / "VINNA/config/checkpoint_paths.yaml"


class CheckpointConfigDict(TypedDict, total=False):
    URL: List[str]
    CKPT: Dict[Plane, Path]
    CFG: Dict[Plane, Path]


def load_checkpoint_config(filename: Union[Path, str] = YAML_DEFAULT) -> CheckpointConfigDict:
    """
    Load the plane dictionary from the yaml file.

    Parameters
    ----------
    filename : Path, str
        Path to the yaml file. Either absolute or relative to the FastSurfer root
        directory.

    Returns
    -------
    CheckpointConfigDict
        A dictionary representing the contents of the yaml file.
    """
    if not filename.absolute():
        filename = VINNA_ROOT / filename

    with open(filename, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    required_fields = ("url", "checkpoint_t1", "checkpoint_t2")
    checks = [k not in data for k in required_fields]
    if any(checks):
        missing = tuple(k for k, c in zip(required_fields, checks) if c)
        message = f"The file {filename} is not valid, missing key(s): {missing}"
        raise IOError(message)
    if isinstance(data["url"], str):
        data["url"] = [data["url"]]
    else:
        data["url"] = list(data["url"])
    for key in ("config", "checkpoint_t1", "checkpoint_t2"):
        if key in data:
            data[key] = {k: Path(v) for k, v in data[key].items()}
    return data


@overload
def load_checkpoint_config_defaults(
        filetype: Literal["checkpoint_t1", "checkpoint_t2", "config"],
        filename: Union[str, Path] = YAML_DEFAULT,
) -> Dict[Plane, Path]: ...


@overload
def load_checkpoint_config_defaults(
        configtype: Literal["url"],
        filename: Union[str, Path] = YAML_DEFAULT,
) -> List[str]: ...


def load_checkpoint_config_defaults(
        configtype: Literal["checkpoint_t1", "checkpoint_t2", "config", "url"],
        filename: Union[str, Path] = YAML_DEFAULT,
) -> Union[Dict[Plane, Path], List[str]]:
    """
    Get the default value for a specific plane or the url.

    Parameters
    ----------
    configtype : "checkpoint_1", "checkpoint_t2", "config", "url
        Type of value.
    filename : str, Path
        The path to the yaml file. Either absolute or relative to the FastSurfer root
        directory.

    Returns
    -------
    List[Plane, Path], List[str]
        Default value for the plane.
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    configtype = configtype.lower()
    if configtype not in ("url", "checkpoint_t1", "checkpoint_t2", "config"):
        raise ValueError("Type must be 'url', 'checkpoint_t1', 'checkpoint_t2' or 'config'")

    return load_checkpoint_config(filename)[configtype]


def create_checkpoint_dir(expr_dir, expr_num, net, aug):
    """
    Create the checkpoint dir if not exists
    :param expr_dir: Base directory to save checkpoints in
    :param expr_num: Name of the experiment
    :return: checkpoint path
    """
    checkpoint_dir = os.path.join(expr_dir, "checkpoints", net, aug, str(expr_num))
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoint(ckpt_dir, epoch):
    checkpoint_dir = os.path.join(ckpt_dir, 'Epoch_{:05d}_training_state.pkl'.format(epoch))
    return checkpoint_dir


def get_checkpoint_path(log_dir, resume_expr_num, net, aug):
    """

    :param log_dir:
    :param resume_expr_num:
    :return:
    """
    if resume_expr_num == "Default":
        return None
    checkpoint_path = os.path.join(log_dir, "checkpoints", net, aug, str(resume_expr_num))
    prior_model_paths = sorted(glob.glob(os.path.join(checkpoint_path, 'Epoch_*')), key=os.path.getmtime)
    if len(prior_model_paths) == 0:
        return None
    return prior_model_paths


def load_from_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, fine_tune=False):
    """
     Loading the model from the given experiment number
    :param checkpoint_path:
    :param model:
    :param optimizer:
    :param scheduler:
    :param fine_tune:
    :return:
        epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    try:
        model.load_state_dict(checkpoint['model_state'])
    except RuntimeError:
        model.module.load_state_dict(checkpoint['model_state'])

    if not fine_tune:
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler and "scheduler_state" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint['epoch']+1, checkpoint['best_metric']


def save_checkpoint(checkpoint_dir, epoch, best_metric, num_gpus, cfg, model,  optimizer, scheduler=None, best=False):
    """
        Saving the state of training for resume or fine-tune
    :param checkpoint_dir:
    :param epoch:
    :param best_metric:
    :param num_gpus:
    :param cfg:
    :param model:
    :param optimizer:
    :param scheduler:
    :return:
    """
    save_name = f"Epoch_{epoch:05d}_training_state.pkl"
    saving_model = model.module if num_gpus > 1 else model
    checkpoint = {
        "model_state": saving_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg.dump()
    }

    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_dir + "/" + save_name)

    if best:
        remove_ckpt(checkpoint_dir + "/Best_training_state.pkl")
        torch.save(checkpoint, checkpoint_dir + "/Best_training_state.pkl")


def remove_ckpt(ckpt):
    try:
        os.remove(ckpt)
    except FileNotFoundError:
        pass

def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: Union[str, Path],
        urls: List[str],
) -> None:
    """
    Download a checkpoint file.

    Raises an HTTPError if the file is not found or the server is not reachable.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint.
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : List[str]
        List of URLs of checkpoint hosting sites.
    """
    response = None
    for url in urls:
        try:
            LOGGER.info(f"Downloading checkpoint {checkpoint_name} from {url} into {checkpoint_path}")
            response = requests.get(url + "/" + checkpoint_name, verify=True)
            # Raise error if file does not exist:
            response.raise_for_status()
            break

        except requests.exceptions.HTTPError as e:
            LOGGER.info(f"Server {url} not reachable.")
            LOGGER.warn(f"Response code: {e.response.status_code}")
        except requests.exceptions.RequestException as e:
            LOGGER.warn(f"Server {url} not reachable.")

    if response is None:
        raise requests.exceptions.RequestException("No server reachable.")
    else:
        response.raise_for_status()  # Raise error if no server is reachable

    with open(checkpoint_path, "wb") as f:
        f.write(response.content)


def check_and_download_ckpts(checkpoint_path: Union[Path, str], urls: List[str]) -> None:
    """
    Check and download a checkpoint file, if it does not exist.

    Parameters
    ----------
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : List[str]
        URLs of checkpoint hosting site.
    """
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
    # Download checkpoint file from url if it does not exist
    if not checkpoint_path.exists():
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls)


def get_checkpoints(*checkpoints: Union[Path, str], urls: List[str]) -> None:
    """
    Check and download checkpoint files if not exist.

    Parameters
    ----------
    *checkpoints : Path, str
        Paths of the files in which the checkpoint will be saved.
    urls : Path, str
        URLs of checkpoint hosting sites.
    """
    try:
        for file in map(Path, checkpoints):
            if not file.is_absolute() and file.parts[0] != ".":
                file = VINNA_ROOT / file
            check_and_download_ckpts(file, urls)
    except requests.exceptions.HTTPError:
        LOGGER.error(f"Could not find nor download checkpoints from {urls}")
        raise