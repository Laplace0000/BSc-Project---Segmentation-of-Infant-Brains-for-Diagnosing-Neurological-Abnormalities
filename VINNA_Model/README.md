# VINNA4neonates
Newborn Whole Brain Segmentation with the Voxel size independent neural network VINNA.

This README contains all information needed to run VINNA4neonates - a fast deep-learning based whole brain segmentation network. 
VINNA4neonates provides a dHCP alternative for volumetric analysis (within minutes) with native sub-millimeter resolution support.

# Getting started

## Installation 
There are two ways to run VINNA4neonates (links are to installation instructions):

1. In a container ([Singularity](doc/overview/INSTALL.md#singularity) or [Docker](doc/overview/INSTALL.md#docker)) (OS: [Linux](doc/overview/INSTALL.md#linux), [Windows](doc/overview/INSTALL.md#windows), [MacOS on Intel](doc/overview/INSTALL.md#docker--currently-only-supported-for-intel-cpus-)),
2. As a [native install](doc/overview/INSTALL.md#native--ubuntu-2004-) (all OS for segmentation part). 

We recommended you use Singularity or Docker, especially if either is already installed on your system, because the 
images we provide on [DockerHub](https://hub.docker.com/r/deepmi/vinna4neonates) conveniently include everything needed for 
VINNA4neonates. We have detailed, per-OS Installation instructions in the [INSTALL.md file](doc/overview/INSTALL.md).

## Usage

All installation methods use the `run_vinna4neonates.sh` call interface (replace `*vinna4neonates-flags*` 
with [VINNA4neonates flags](doc/overview/FLAGS.md#required-arguments)), which is the general starting point for VINNA4neonates. 
However, there are different ways to call this script depending on the installation, which we explain here:

1. For container installations, you need to define the hardware and mount the folders with the input (`/data`) and output data (`/output`):  
   (a) For __singularity__, the syntax is 
    ```
    singularity exec --nv \
                     --no-home \
                     -B /home/user/my_mri_data:/data \
                     -B /home/user/my_vinna4neonates_analysis:/output \
                     ./vinna4neonates-gpu.sif \
                     /vinna4neonates/run_vinna4neonates.sh 
                     *vinna4neonates-flags*
   ```
   The `--nv` flag is needed to allow VINNA4neonates to run on the GPU (otherwise VINNA4neonates will run on the CPU).

   The `--no-home` flag tells singularity to not mount the home directory (see [Singularity README](Singularity/README.md#mounting-home) for more info).

   The `-B` flag is used to tell singularity, which folders VINNA4neonates can read and write to.
 
   See also __[Example 2](doc/overview/EXAMPLES.md#example-2--vinna4neonates-singularity)__ for a full singularity 
   VINNA4neonates run command.  

   (b) For __docker__, the syntax is
    ```
    docker run --gpus all \
               -v /home/user/my_mri_data:/data \
               -v /home/user/my_vinna4neonates_analysis:/output \
               --rm --user $(id -u):$(id -g) \
               deepmi/vinna4neonates:latest \
               *vinna4neonates-flags*
    ```
   The `--gpus` flag is needed to allow VINNA4neonates to run on the GPU (otherwise VINNA4neonates will run on the CPU).

   The `-v` flag is used to tell docker, which folders VINNA4neonates can read and write to.
 
   See also __[Example 1](doc/overview/EXAMPLES.md#example-1--vinna4neonates-docker)__ for a full VINNA4neonates run 
   inside a Docker container and [the Docker README](Docker/README.md#docker-flags-) for more details on the docker flags including `--rm` and `--user`.

   For a __native install__, you need to activate your VINNA4neonates environment (e.g. `conda activate vinna4neonates_gpu`)
   and make sure you have added the VINNA4neonates path to your `PYTHONPATH` variable, e.g. `export PYTHONPATH=$(pwd)`. 

   You will then be able to run vinna4neonates with `./run_vinna4neonates.sh *vinna4neonates-flags*`.

   See also [Example 3](doc/overview/EXAMPLES.md#example-3--native-vinna4neonates-on-subjectx-) 
   for an illustration of the commands to run VINNA4neonates natively.
