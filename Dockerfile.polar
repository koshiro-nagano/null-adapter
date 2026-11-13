# syntax=docker/dockerfile:1
# polar/ image: SimCMF's SAM-B + LoRA on an RTX 5080 (sm_120 needs torch >= 2.7, CUDA 12.8 wheels; host driver >= 570).
# Versions are those of the .venv the experiments were first run in.
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# the proxy re-signs TLS: build_docker.sh passes the host CA bundle as the optional build secret "ca"
RUN --mount=type=secret,id=ca \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi \
 && pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 \
 && pip install numpy==1.26.4 monai==1.3.1 einops==0.8.2 opencv-python-headless==4.11.0.86 \
      pillow==13.0.0 tensorboardX==2.6.5 tqdm==4.70.1 pandas==3.0.6 scipy==1.17.1 pyyaml==6.0.3 pytest==9.2.0
# SimCMF (no license file, not vendored) is mounted read-only at /simcmf by run_docker.sh
ENV SIMCMF_ROOT=/simcmf
WORKDIR /work
