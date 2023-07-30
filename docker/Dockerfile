FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV WORK_PATH /workspace

RUN apt-get update && \
    apt-get install -y \
			sudo curl git build-essential python3-pip && \
    apt-get clean && \
		rm -rf /var/lib/apt/lists/* && \
		rm -rf /var/cache/apk/

COPY requirements.txt .
ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
RUN pip3 install -U pip && \
    pip3 install -r requirements.txt

# user
COPY entrypoint.sh /entrypoint.sh
RUN chmod 777 /entrypoint.sh

ARG USER_NAME=duser
ARG GROUP_NAME=build
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid $GROUP_ID -o $GROUP_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME && \
    adduser $USER_NAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USER_NAME

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh", "bash"]