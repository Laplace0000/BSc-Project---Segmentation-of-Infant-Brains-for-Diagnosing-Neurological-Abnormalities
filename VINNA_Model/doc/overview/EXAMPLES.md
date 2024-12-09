# Examples

## Example 1: VINNA4neonates Docker
After pulling one of our images from Dockerhub, you do not need to have a separate installation of FreeSurfer on your 
computer (it is already included in the Docker image). Basically for Docker (as for Singularity below) you are starting 
a container image (with the run command) and pass several parameters for that, e.g. if GPUs will be used and mounting 
(linking) the input and output directories to the inside of the container image. In the second half of that call you 
pass parameters to our run_vinna4neonates.sh script that runs inside the container (e.g. where to find the input data 
and other flags). 

To run VINNA4neonates on a given subject using the provided GPU-Docker, execute the following command:

```bash
# 1. get the vinna4neonates docker image (if it does not exist yet)
docker pull deepmi/vinna4neonates 

# 2. Run command
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_vinna4neonates_analysis:/output \
                      --rm --user $(id -u):$(id -g) deepmi/vinna4neonates:latest \
                      --t2 /data/subjectX/t2-weighted.nii.gz \
                      --sid subjectX --sd /output 
```

Docker Flags:
* The `--gpus` flag is used to allow Docker to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus 'device=0,1,3'`.
* The `-v` commands mount your data, output, and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 
* The `--rm` flag takes care of removing the container once the analysis finished. 
* The `--user $(id -u):$(id -g)` part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root which is discouraged.

VINNA4neonates Flag:
* The `--t2` points to the t2-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_vinna4neonates_analysis => /output)

Note, that the paths following `--t1` and `--sd` are __inside__ the container, not global paths on your system, so they 
should point to the places where you mapped these paths above with the `-v` arguments (part after colon). 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not 
exist. So in this example output will be written to /home/user/my_vinna4neonates_analysis/subjectX/ . Make sure the 
output directory is empty, to avoid overwriting existing files. 

If you do not have a GPU, you can also run our CPU-Docker by dropping the `--gpus all` flag and specifying `--device cpu`
at the end as a VINNA4neonates flag. See [Docker/README.md](Docker/README.md) for more details.

## Example 2: VINNA4neonates Singularity
After building the Singularity image (see below or instructions in ./Singularity/README.md), you may run VINNA4neonates 
on a given subject using the Singularity image with GPU access, execute the following commands from a directory where 
you want to store singularity images. This will create a singularity image from our Dockerhub image and execute it:

```bash
# 1. Build the singularity image (if it does not exist)
singularity build vinna4neonates-gpu.sif docker://deepmi/vinna4neonates

# 2. Run command
singularity exec --nv \
                 --no-home \
                 -B /home/user/my_mri_data:/data \
                 -B /home/user/my_vinna4neonates_analysis:/output \
                 -B /home/user/my_fs_license_dir:/fs_license \
                 ./vinna4neonates-gpu.sif \
                 /vinna4neonates/run_vinna4neonates.sh \
                 --t2 /data/subjectX/t2-weighted.nii.gz \
                 --sid subjectX --sd /output 
```

### Singularity Flags
* The `--nv` flag is used to access GPU resources. 
* The `--no-home` flag stops mounting your home directory into singularity.
* The `-B` commands mount your data, output, and directory with the FreeSurfer license file into the Singularity container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 

### VINNA4neonates Flags
* The `--t2` points to the t2-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_vinna4neonates_analysis => /output)

Note, that the paths following `--t2` and `--sd` are __inside__ the container, not global paths on your system, so they 
should point to the places where you mapped these paths above with the `-v` arguments (part after colon).

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory. So in this 
example output will be written to /home/user/my_vinna4neonates_analysis/subjectX/ . Make sure the output directory is 
empty, to avoid overwriting existing files. 

You can run the Singularity equivalent of CPU-Docker by building a Singularity image from the CPU-Docker image and 
excluding the `--nv` argument in your Singularity exec command. Also append `--device cpu` as a VINNA4neonates flag.


## Example 3: Native VINNA4neonates on subjectX 

For a native install you may want to make sure that you are on our stable branch, as the default dev branch is for 
development and could be broken at any time. For that you can directly clone the stable branch:

```bash
git clone --branch stable https://github.com/Deep-MI/VINNA4neonates.git
```

More details (e.g. you need all dependencies in the right versions and also FastSurfer locally) can be found in our [INSTALL.md file](INSTALL.md).
Given you want to analyze data for subject which is stored on your computer under 
/home/user/my_mri_data/subjectX/t1-weighted.nii.gz, run the following command from the console:

```bash
# Define data directory
datadir=/home/user/my_mri_data
vinna4neonatesdir=/home/user/my_vinna4neonates_analysis

# Run VINNA4neonates
./run_vinna4neonates.sh --t2 $datadir/subjectX/t2-weighted-nii.gz \
                    --sid subjectX --sd $vinna4neonatesdir 
```

The output will be stored in the $vinna4neonatesdir.