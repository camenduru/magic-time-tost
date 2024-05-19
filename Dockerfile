FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip && \
    pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av xformers==0.0.25 einops omegaconf accelerate==0.28.0 \
    diffusers==0.11.1 transformers==4.38.2 jax==0.4.19 jaxlib==0.4.19 ms-swift

RUN GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/camenduru/MagicTime /content/MagicTime && \
    git clone -b fp16 https://huggingface.co/runwayml/stable-diffusion-v1-5 /content/MagicTime/ckpts/Base_Model/stable-diffusion-v1-5 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/Base_Model/motion_module/motion_module.ckpt -d /content/MagicTime/ckpts/Base_Model/motion_module -o motion_module.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/DreamBooth/RcnzCartoon.safetensors -d /content/MagicTime/ckpts/DreamBooth -o RcnzCartoon.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/DreamBooth/RealisticVisionV60B1_v51VAE.safetensors -d /content/MagicTime/ckpts/DreamBooth -o RealisticVisionV60B1_v51VAE.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/DreamBooth/ToonYou_beta6.safetensors -d /content/MagicTime/ckpts/DreamBooth -o ToonYou_beta6.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/Magic_Weights/magic_adapter_s/magic_adapter_s.ckpt -d /content/MagicTime/ckpts/Magic_Weights/magic_adapter_s -o magic_adapter_s.ckpt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/raw/main/ckpts/Magic_Weights/magic_adapter_t/configuration.json -d /content/MagicTime/ckpts/Magic_Weights/magic_adapter_t -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/raw/main/ckpts/Magic_Weights/magic_adapter_t/default/adapter_config.json -d /content/MagicTime/ckpts/Magic_Weights/magic_adapter_t/default -o adapter_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/Magic_Weights/magic_adapter_t/default/adapter_model.bin -d /content/MagicTime/ckpts/Magic_Weights/magic_adapter_t/default -o adapter_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/raw/main/ckpts/Magic_Weights/magic_text_encoder/configuration.json -d /content/MagicTime/ckpts/Magic_Weights/magic_text_encoder -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/resolve/main/ckpts/Magic_Weights/magic_text_encoder/default/adapter_model.bin -d /content/MagicTime/ckpts/Magic_Weights/magic_text_encoder/default -o adapter_model.bin && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/spaces/BestWishYsh/MagicTime/raw/main/ckpts/Magic_Weights/magic_text_encoder/default/adapter_config.json -d /content/MagicTime/ckpts/Magic_Weights/magic_text_encoder/default -o adapter_config.json
COPY ./worker_runpod.py /content/MagicTime/worker_runpod.py
USER camenduru
WORKDIR /content/MagicTime
CMD python worker_runpod.py