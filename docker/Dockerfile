ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=project
ARG USER_NAME=ubuntu
ARG GROUP_NAME=ubuntu
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.11
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"
ARG FORCE_CUDA=0
# https://developer.nvidia.com/cuda-gpus
ARG TORCH_CUDA_ARCH_LIST=7.5

ENV DEBIAN_FRONTEND="noninteractive" \
    LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    PYTHONPATH=${APPLICATION_DIRECTORY}
ENV FORCE_CUDA=${FORCE_CUDA}

# Following is needed to install python 3.7
RUN apt update && apt install --no-install-recommends -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt update && apt install --no-install-recommends -y \
    git curl make ssh openssh-client \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-distutils python3-pip python-is-python3

# Following is needed to switch default python3 version
# For detail, please check following link https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
# `requests` needs to be upgraded to avoid RequestsDependencyWarning
# ref: https://stackoverflow.com/questions/56155627/requestsdependencywarning-urllib3-1-25-2-or-chardet-3-0-4-doesnt-match-a-s
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN python3 -m pip install --upgrade pip setuptools requests

# basic libs
RUN apt install -y build-essential
RUN apt install -y ffmpeg
RUN python3 -m pip install -U pip
RUN python3 -m pip install scikit-image matplotlib imageio opencv-python
RUN python3 -m pip install plotly pandas
RUN python3 -m pip install tqdm
RUN python3 -m pip install loguru
RUN python3 -m pip install tensorboard

# pytorch (https://pytorch.org/get-started/previous-versions/)
RUN python3 -m pip install torch torchvision

# Clean
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}
