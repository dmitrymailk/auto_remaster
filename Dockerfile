# FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
# FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
# seg fault
# FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel 
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel
ARG DEBIAN_FRONTEND=noninteractive
# https://github.com/docker/build-push-action/issues/933#issuecomment-1687372123
RUN rm /etc/apt/sources.list.d/cuda*.list
# RUN  apt-get upgrade -y && apt-get install ffmpeg libsm6 libxext6 ncdu -y
RUN apt-get update -y --fix-missing && apt-get install ffmpeg libsm6 libxext6 ncdu -y --no-install-recommends
RUN apt-get install git curl numactl wget unzip iproute2 htop git-lfs -y --no-install-recommends && pip install nvitop 

ARG USERNAME=user-name-goes-here
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    # && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
RUN  echo -e "\nexport PATH=$PATH:/home/user-name-goes-here/.local/bin\n" >>  /home/user-name-goes-here/.bashrc 
# RUN sudo chown -R $USERNAME /opt/conda
RUN conda init
WORKDIR /code
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y