FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    git \
    vim \
    screen \
    && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install torch==1.9.0+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN python3.9 -m pip install guildai gpustat

COPY requirements.txt /requirements.txt
RUN python3.9 -m pip install --no-cache-dir -r /requirements.txt

CMD ['bash']