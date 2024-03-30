FROM jupyter/base-notebook:aarch64-2023-06-26 AS base-builder

USER root

RUN sed -i "s@http://ports.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y build-essential cmake git libhdf5-dev

RUN pip config --global set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    conda init --system && \
    conda config --system --add default_channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2' && \
    conda config --system --add default_channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' && \
    conda config --system --add default_channels 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' && \
    conda config --system --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.msys2 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.bioconda https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.menpo https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.pytorch-lts https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --system --set custom_channels.simpleitk https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
COPY ./build/jupyter_lab_config.py /etc/jupyter

FROM base-builder AS conda-builder

USER jovyan

# install conda environments
COPY --chown=jovyan:users build/environments ./build/environments
COPY --chown=jovyan:users build/create_envs.sh ./build/create_envs.sh
RUN --mount=type=cache,target=/opt/conda/pkgs,uid=1000,gid=1000 \
    --mount=type=cache,target=/home/jovyan/.cache/pip,uid=1000,gid=100 \
    --mount=type=cache,target=/home/jovyan/.cache/conda,uid=1000,gid=100 \
    bash ./build/create_envs.sh ./build/environments

FROM base-builder

USER jovyan
WORKDIR /home/jovyan

# install jupyterlab extensions or dependencies
# COPY --chown=jovyan:users ./build/requirements.txt ./build/requirements.txt

# RUN --mount=type=cache,target=/home/jovyan/.cache/pip \
#     pip install -r ./build/requirements.txt

COPY --from=conda-builder --chown=jovyan:users --chmod=2775 /opt/conda/envs /opt/conda/envs
COPY --from=conda-builder --chown=jovyan:users --chmod=2775 /opt/conda/share /opt/conda/share

