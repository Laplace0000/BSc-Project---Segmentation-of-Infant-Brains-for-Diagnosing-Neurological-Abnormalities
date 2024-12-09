FROM ubuntu:20.04 AS build
ENV LANG=C.UTF-8
ARG PYTHON_VERSION=3.8
ARG CONDA_FILE=Miniconda3-py38_4.11.0-Linux-x86_64.sh
ENV DEBIAN_FRONTEND=noninteractive

# Install packages needed for build
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      upx \
      file && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone FastSurfer (conform script and entry point used)
RUN git clone --depth 1 --branch v2.2.0 https://github.com/Deep-MI/FastSurfer.git /FastSurfer

# Install conda
RUN wget --no-check-certificate -qO ~/miniconda.sh https://repo.continuum.io/miniconda/$CONDA_FILE  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh

ENV PYTHONUNBUFFERED=0 \
    PATH=/opt/conda/bin:$PATH

# Install our dependencies
COPY ./Docker/vinna4neonates_env_gpu.yaml /VINNA4neonates/vinna4neonates_env_gpu.yaml
RUN conda env create -f /VINNA4neonates/vinna4neonates_env_gpu.yaml

# Install conda-pack:
RUN conda install -c conda-forge conda-pack

# Use conda-pack to create a standalone enviornment in /venv:
RUN conda-pack -n vinna4neonates_gpu -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# Now that venv in a new location, fix up paths:
RUN /venv/bin/conda-unpack
ENV PATH /venv/bin:$PATH

# ========================================
# Here we create the smaller runtime image
# ========================================

FROM ubuntu:20.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      git \
      ca-certificates \
      upx \
      file && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Add venv to path:
ENV PATH=/venv/bin:$PATH

# make sure we use bash and activate conda env
#  (in case someone starts this interactively)
RUN echo "source /venv/bin/activate" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy venv and FastSurferCNN
COPY --from=build /venv /venv
COPY --from=build /FastSurfer /FastSurfer
COPY ./VINNA/ /VINNA4neonates/VINNA
COPY ./run_vinna4neonates.sh /VINNA4neonates/run_vinna4neonates.sh

# Set all Pythonpath references already
ENV PYTHONPATH=/FastSurfer:/VINNA4neonates:$PYTHONPATH \
    MPLCONFIGDIR=/tmp \
    FASTSURFER_HOME=/FastSurfer \
    VINNA4NEONATES_HOME=/VINNA4neonates
 
# Download remote network checkpoints
RUN cd /VINNA4neonates ; python3 VINNA/download_checkpoints.py --vinna
   
# Set Workdir and entrypoint
#  the script entrypoint ensures that our conda env is active
WORKDIR "/VINNA4neonates"
USER nonroot
ENTRYPOINT ["/FastSurfer/Docker/entrypoint.sh", "bash", "/VINNA4neonates/run_vinna4neonates.sh"]

CMD ["--help"]
