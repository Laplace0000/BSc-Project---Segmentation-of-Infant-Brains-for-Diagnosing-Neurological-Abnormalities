# VINNA4neonates Docker Image Creation

Within this directory we currently provide a Dockerfiles for the segmentation with the VINNA4neonates.
Note, in order to run our Docker containers on a Mac, users need to increase docker memory to 10 GB by overwriting the settings under Docker Desktop --> Preferences --> Resources --> Advanced (slide the bar under Memory to 10 GB; see: [docker for mac](https://docs.docker.com/docker-for-mac/) for details).  

### Build GPU VINNA4neonates Image

In order to build your own Docker image for VINNA4neonates simply execute the following command after traversing into the *Docker* directory: 

```bash
cd ..
docker build --rm=true -t vinna4neonates:latest -f ./Docker/Dockerfile .
```

After building or pulling the VINNA4neonates image from Dockerhub, you do not need to have a separate installation of 
FastSurfer on your computer (it is already included in the Docker image). Basically for Docker you are starting 
a container image (with the run command) and pass several parameters for that, e.g. if GPUs will be used and mounting 
(linking) the input and output directories to the inside of the container image. In the second half of that call you 
pass parameters to our run_vinna4neonates.sh script that runs inside the container (e.g. where to find the input data 
and other flags). 

### VINNA4neonates on T2-weighted image
To run VINNA4neonates on a given subject with a T2-weighted image using the provided GPU-Docker, execute the following command:

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_vinna4neonates_analysis:/output \
                      --rm --user $(id -u):$(id -g) deepmi/vinna4neonates:latest \
                      --t2 /data/subjectX/t2-weighted.nii.gz \
                      --sid subjectX_T2 --sd /output 
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

Note, that the paths following `--t2` and `--sd` are __inside__ the container, not global paths on your system, so they 
should point to the places where you mapped these paths above with the `-v` arguments (part after colon). 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not 
exist. So in this example output will be written to /home/user/my_vinna4neonates_analysis/subjectX/ . Make sure the 
output directory is empty, to avoid overwriting existing files. 

If you do not have a GPU, you can also run our CPU-Docker by dropping the `--gpus all` flag and specifying `--device cpu`
at the end as a VINNA4neonates flag.

### VINNA4neonates on T1-weighted image
To run VINNA4neonates on a given subject with a T1-weighted image using the provided GPU-Docker, simply change the
--t2 to the --t1 flag. All other flags stay the same:

```bash
docker run --gpus all -v /home/user/my_mri_data:/data \
                      -v /home/user/my_vinna4neonates_analysis:/output \
                      --rm --user $(id -u):$(id -g) deepmi/vinna4neonates:latest \
                      --t1 /data/subjectX/t1-weighted.nii.gz \
                      --sid subjectX_T1 --sd /output 
```

Docker Flags:
* The `--gpus` flag is used to allow Docker to access GPU resources. With it, you can also specify how many GPUs to use. In the example above, _all_ will use all available GPUS. To use a single one (e.g. GPU 0), set `--gpus device=0`. To use multiple specific ones (e.g. GPU 0, 1 and 3), set `--gpus 'device=0,1,3'`.
* The `-v` commands mount your data, output, and directory with the FreeSurfer license file into the docker container. Inside the container these are visible under the name following the colon (in this case /data, /output, and /fs_license). 
* The `--rm` flag takes care of removing the container once the analysis finished. 
* The `--user $(id -u):$(id -g)` part automatically runs the container with your group- (id -g) and user-id (id -u). All generated files will then belong to the specified user. Without the flag, the docker container will be run as root which is discouraged.

VINNA4neonates Flag:
* The `--t1` points to the t1-weighted MRI image to analyse (full path, with mounted name inside docker: /home/user/my_mri_data => /data)
* The `--sid` is the subject ID name (output folder name)
* The `--sd` points to the output directory (its mounted name inside docker: /home/user/my_vinna4neonates_analysis => /output)

Note, that the paths following `--t1` and `--sd` are again __inside__ the container, not global paths on your system, so they 
should point to the places where you mapped these paths above with the `-v` arguments (part after colon). 

A directory with the name as specified in `--sid` (here subjectX) will be created in the output directory if it does not 
exist. So in this example output will be written to /home/user/my_vinna4neonates_analysis/subjectX/ . Make sure the 
output directory is empty, to avoid overwriting existing files. 

If you do not have a GPU, you can also run our CPU-Docker by dropping the `--gpus all` flag and specifying `--device cpu`
at the end as a VINNA4neonates flag.