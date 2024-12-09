# Installation

VINNA4neonates is a network for the segmentation of human newborn brain MRI data. 

The preferred way of installing and running VINNA4neonates is via Singularity or Docker containers. 
We provide pre-build images at Dockerhub. 

We also provide information on a native install on some operating systems, but since dependencies may vary, this can 
produce results different from our testing environment and we may not be able to support you if things don't work. 
Our testing is performed on Ubuntu 20.04 via our provided Docker images.


## Linux

Recommended System Spec: 8 GB system memory, NVIDIA GPU with 8 GB graphics memory.

Minimum System Spec: 8 GB system memory (this requires running VINNA4neonates on the CPU only, which is much slower)

### Singularity

Assuming you have singularity installed already (by a system admin), you can build an image easily from our Dockerhub 
images. Run this command from a directory where you want to store singularity images:

```bash
singularity build vinna4neonates-gpu.sif docker://deepmi/vinna4neonates:latest
```
Additionally, [the Singularity README](Singularity/README.md) contains detailed directions for building your own 
Singularity images from Docker.

Our [README](README.md#example-2--vinna4neonates-singularity) explains how to run VINNA4neonates and you can find 
details on how to build your own images here: [Docker](Docker/README.md) and [Singularity](Singularity/README.md). 


### Docker

This is very similar to Singularity. Assuming you have Docker installed (by a system admin) you just need to pull one of 
our pre-build Docker images from dockerhub:

```bash
docker pull deepmi/vinna4neonates:latest
```

Our [README](README.md#example-1--vinna4neonates-docker) explains how to run VINNA4neonates and you can find details 
on how to [build your own image](Docker/README.md). 


### Native (Ubuntu 20.04 or Ubuntu 22.04)

In a native install you need to install all dependencies (distro packages, python dependencies, FastSurfer repo) 
yourself. Here we will walk you through what you need.

#### 1. System Packages

You will need a few additional packages that may be missing on your system (for this you need sudo access or ask a system admin):

```bash
sudo apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      file
```

If you are using **Ubuntu 20.04**, you will need to upgrade to a newer version of libstdc++, as some 'newer' python 
packages need GLIBCXX 3.4.29, which is not distributed with Ubuntu 20.04 by default.

```bash
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y g++-11
```

You also need to have bash-4.0 or higher (check with `bash --version`). 

You also need a working version of python3 (we recommend python 3.9 -- we do not support other versions). These 
packages should be sufficient to install python dependencies and then run the VINNA4neonates neural network segmentation. 

If you are using pip, make sure pip is updated as older versions will fail.

#### 2. Conda for python

We recommend to install conda as your python environment. If you don't have conda on your system, an admin needs to install it:

```bash
wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
chmod +x ~/miniconda.sh
sudo ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh 
```

#### 3. FastSurfer
Get FastSurfer from GitHub into your specified directory $BASEDIR. Here you can decide if you want to install the 
current experimental "dev" version (which can be broken), the "stable" branch (latest version that has been tested 
thoroughly), or a specific tag (which we use here --> tag v2.2.0):

```bash
cd $BASEDIR
RUN git clone --depth 1 --branch v2.2.0 https://github.com/Deep-MI/FastSurfer.git 
```

#### 4. VINNA4neonates
Get VINNA4neonates from GitHub. Here you can decide if you want to install the current experimental "dev" version 
(which can be broken) or the "stable" branch (that has been tested thoroughly):

```bash
git clone --branch stable https://github.com/Deep-MI/VINNA4neonates.git
cd VINNA4neonates
```

#### 5. Python environment

Create a new environment and install VINNA4neonates dependencies:

```bash
conda env create -f ./env/vinna4neonates.yml 
conda activate vinna4neonates
```

Next, add the VINNA4neonates and FastSurfer directory to the python path 
(make sure you have changed into VINNA4neonates already):
```bash
export PYTHONPATH="${PYTHONPATH}:$PWD:$BASEDIR/FastSurfer"
```

This will need to be done every time you want to run VINNA4neonates, or you need to add this line to your `~/.bashrc` 
if you are using bash, for example:
```bash
echo "export PYTHONPATH=\"\${PYTHONPATH}:$PWD:$BASEDIR/FastSurfer\"" >> ~/.bashrc
```

You can also download all network checkpoint files (this should be done if you are installing for multiple users):
```bash
python3 VINNA4neonatesCNN/download_checkpoints.py --vinna
```

Once all dependencies are installed, you are ready to run the VINNA4neonates by calling 
```./run_vinna4neonates.sh ....``` , see the [README](README.md#example-3--native-vinna4neonates-on-subjectx-) for 
command line flags.

## MacOS
### Docker (currently only supported for Intel CPUs)

Docker can be used on Intel Macs as it should be similarly fast as a native install there. 
It would allow you to run the full pipeline.

First, install [Docker Desktop for Mac](https://docs.docker.com/get-docker/).
Start it and set Memory to 15 GB under Preferences -> Resources (or the largest you have, if you are below 15GB, it may fail). 

Second, pull one of our Docker containers. Open a terminal window and run:

```sh
docker pull deepmi/vinna4neonates:latest
```

Continue with the example in our [README](README.md#example-1--vinna4neonates-docker). 


### Native

On modern Macs with the Apple Silicon M1 or M2 ARM-based chips, we recommend a native installation as it runs much 
faster than Docker in our tests. The experimental support for the built-in AI Accelerator is also only available on 
native installations. Native installation also supports older Intel chips.

#### 1. Git and Bash
If you do not have git and a recent bash (version > 4.0 required!) installed, install them via the packet manager, e.g. brew.
This installs brew and then bash:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install bash
```

Make sure you use this bash and not the older one provided with MacOS!

#### 2. Python
Create a python environment, activate it, and upgrade pip. Here we use pip, but you should also be able to use conda for python: 

```sh
python3 -m venv $HOME/python-envs/vinna4neonates 
source $HOME/python-envs/vinna4neonates/bin/activate
python3 -m pip install --upgrade pip
```

#### 3. FastSurfer
Get FastSurfer from GitHub into your specified directory $BASEDIR. Here you can decide if you want to install the 
current experimental "dev" version (which can be broken), the "stable" branch (latest version that has been tested 
thoroughly), or a specific tag (which we use here --> tag v2.2.0):

```bash
cd $BASEDIR
RUN git clone --depth 1 --branch v2.2.0 https://github.com/Deep-MI/FastSurfer.git 
cd FastSurfer
export PYTHONPATH="${PYTHONPATH}:$PWD
```

#### 4. VINNA4neonates and Requirements
Clone VINNA4neonates:
```sh
git clone --branch stable https://github.com/Deep-MI/VINNA4neonates.git
cd VINNA4neonates
export PYTHONPATH="${PYTHONPATH}:$PWD"
```


Install the VINNA4neonates requirements
```sh
python3 -m pip install -r requirements.mac.txt
```

If this step fails, you may need to edit ```requirements.mac.txt``` and adjust version number to what is available. 
On newer M1 Macs, we also had issues with the h5py package, which could be solved by using brew for help (not sure this is needed any longer):

```sh
brew install hdf5
export HDF5_DIR="$(brew --prefix hdf5)"
pip3 install --no-binary=h5py h5py
```

You can also download all network checkpoint files (this should be done if you are installing for multiple users):
```sh
python3 VINNA4neonatesCNN/download_checkpoints.py --all
```

Once all dependencies are installed, run the VINNA4neonates by calling ```bash ./run_vinna4neonates.sh ....``` 
with the appropriate command line flags, see the [README](README.md#usage). 

Note: You may always need to prepend the command with `bash` (i.e. `bash run_vinna4neonates.sh <...>`) to ensure that 
bash 4.0 is used instead of the system default.

## Windows

### Docker (CPU version)

In order to run VINNA4neonates on your Windows system using docker make sure that you have:
* [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)
* [Docker Desktop](https://docs.docker.com/desktop/install/windows-install/)

installed and running.

After everything is installed, start Windows PowerShell and run the following command to pull the CPU Docker image 
(check on [dockerhub](https://hub.docker.com/r/deepmi/vinna4neonates/tags) what version tag is most recent for cpu):

```bash
docker pull deepmi/vinna4neonates:latest
```

Now you can run VINNA4neonates the same way as described in our [README](README.md#example-1--vinna4neonates-docker) for
the CPU build, for example:
```bash
docker run -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_vinna4neonates_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/vinna4neonates:latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --device cpu \
           --sid subjectX --sd /output \
           --parallel
```
Note, the [system requirements](https://github.com/Deep-MI/VINNA4neonates#system-requirements) of at least 8GB of RAM 
for the CPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).

This was tested using Windows 10 Pro version 21H1 and the WSL Ubuntu 20.04  distribution

### Docker (GPU version)

In addition to the requirements from the CPU version, you also need to make sure that you have:
* Windows 11 or Windows 10 21H2 or greater,
* the latest WSL Kernel or at least 4.19.121+ (5.10.16.3 or later for better performance and functional fixes),
* an NVIDIA GPU and the latest [NVIDIA CUDA driver](https://developer.nvidia.com/cuda/wsl)
* CUDA toolkit installed on WSL, see: _[CUDA Support for WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2)_

Follow [Enable NVIDIA CUDA on WSL](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl) to install the correct drivers and software.

After everything is installed, start Windows PowerShell and run the following command to pull the GPU Docker image:

```bash
docker pull deepmi/vinna4neonates:latest
```

Now you can run VINNA4neonates the same way as described in our [README](README.md#example-1--vinna4neonates-docker), for example:
```bash
docker run --gpus all
           -v C:/Users/user/my_mri_data:/data \
           -v C:/Users/user/my_vinna4neonates_analysis:/output \
           -v C:/Users/user/my_fs_license_dir:/fs_license \
           --rm --user $(id -u):$(id -g) deepmi/vinna4neonates:latest \
           --fs_license /fs_license/license.txt \
           --t1 /data/subjectX/orig.mgz \
           --sid subjectX --sd /output \
           --parallel
```

Note the [system requirements](https://github.com/Deep-MI/VINNA4neonates#system-requirements) of at least 8 GB system 
memory and 2 GB graphics memory for the GPU version. If the process fails, check if your [WSL2 distribution has enough memory reserved](https://www.aleksandrhovhannisyan.com/blog/limiting-memory-usage-in-wsl-2/).
