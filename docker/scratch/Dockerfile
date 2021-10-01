# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.4.0-cudnn8-runtime-ubuntu20.04

# Basic setup
RUN DEBIAN_FRONTEND=noninteractive apt-get update -yy \
    && apt-get install -yy --no-install-recommends python3 python3-pip ffmpeg libsm6 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# If I don't separate these out, then it asks for the timezone...
RUN TZ=Europe/Vienna DEBIAN_FRONTEND=noninteractive apt-get update -yy \
    && apt-get install -yy openssh-client git\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# Basic requirements
RUN mkdir /tmp
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# Private packages from git
# From: https://medium.com/@tonistiigi/build-secrets-and-ssh-forwarding-in-docker-18-09-ae8161d066
# download public key for github.com

RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN --mount=type=ssh echo "Using ssh keys: $(ssh-add -L)"
RUN --mount=type=ssh echo "$(ssh -T git@github.com)"

# clone our private repositories
RUN --mount=type=ssh git clone git@github.com:Zimmer-lab/dlc_for_wbfm.git dlc_for_wbfm
RUN pip install --no-cache-dir --upgrade ./dlc_for_wbfm

RUN --mount=type=ssh git clone git@github.com:Zimmer-lab/segmentation.git segmentation
RUN pip install --no-cache-dir --upgrade ./segmentation

# Direct pip installation gives permission errors...
#RUN --mount=type=ssh pip install git+ssh://git@github.com/Zimmer-lab/dlc_for_wbfm.git
#RUN --mount=type=ssh python3 -m pip install git+ssh://github.com/Zimmer-lab/segmentation.git

# Run tests of the installation
