FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install tzdata && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install guildai
RUN python3.9 -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

COPY requirements.txt /recurrent_graph_vae/requirements.txt
WORKDIR recurrent_graph_vae

RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install --no-cache-dir -r ./requirements.txt

CMD ['bash']
