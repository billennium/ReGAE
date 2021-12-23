FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install tzdata && \
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

RUN python3.9 -m pip install torch torchvision torchaudio
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install guildai 

COPY requirements.txt /requirements.txt
RUN python3.9 -m pip install --no-cache-dir -r /requirements.txt

CMD ['bash']
