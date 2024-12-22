ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

ARG PROJECT_NAME=project
ARG USER_NAME=challenger
ARG GROUP_NAME=challengers
ARG UID=1000
ARG GID=1000
ARG PYTHON_VERSION=3.8
ARG APPLICATION_DIRECTORY=/home/${USER_NAME}/${PROJECT_NAME}
ARG RUN_POETRY_INSTALL_AT_BUILD_TIME="false"
ARG FORCE_CUDA=0
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

# Following is needed to swtich default python3 version
# For detail, please check following link https://unix.stackexchange.com/questions/410579/change-the-python3-default-version-in-ubuntu
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
# `requests` needs to be upgraded to avoid RequestsDependencyWarning
# ref: https://stackoverflow.com/questions/56155627/requestsdependencywarning-urllib3-1-25-2-or-chardet-3-0-4-doesnt-match-a-s
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
RUN python3 -m pip install --upgrade pip setuptools requests

# basic libs
RUN apt install -y build-essential
RUN python3 -m pip install -U pip
RUN python3 -m pip install scikit-image matplotlib imageio opencv-python
RUN python3 -m pip install plotly pandas
RUN python3 -m pip install tqdm
RUN python3 -m pip install loguru
RUN python3 -m pip install tensorboard

# pytorch
RUN python3 -m pip install torch torchvision

# Add user. Without this, following process is executed as admin.
RUN groupadd -g ${GID} ${GROUP_NAME} \
    && useradd -ms /bin/sh -u ${UID} -g ${GID} ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${APPLICATION_DIRECTORY}

# # If ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true", install Python package by Poetry and move .venv under ${HOME}.
# # This process is for CI (GitHub Actions). To prevent overwrite by volume of docker compose, .venv is moved under ${HOME}.
# COPY --chown=${UID}:${GID} pyproject.toml poetry.lock poetry.toml .
# RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && poetry install || echo "skip to run poetry install."
# RUN test ${RUN_POETRY_INSTALL_AT_BUILD_TIME} = "true" && mv ${APPLICATION_DIRECTORY}/.venv ${HOME}/.venv || echo "skip to move .venv."
