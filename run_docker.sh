#!/usr/bin/env bash
# Usage: ./run_docker.sh <command...>   (runs inside null-adapter/ as /work, as host user)
# VARIANT=cpu to use the CPU image, VARIANT=phase2 for the DINOv3/SAM 3 image (passes only HF_TOKEN from .env); GPUS=all|none|"device=0" (default all with the gpu image)
# VARIANT=polar for polar/ (SimCMF clone mounted read-only at /simcmf from SIMCMF_ROOT, default ../ext/SimCMF)
set -e
cd "$(dirname "$0")"
variant="${VARIANT:-gpu}"
gpu_flags=()
hf_flags=()
image="null-adapter:toy1-$variant"
if [ "$variant" = phase2 ]; then
  image=null-adapter:phase2
  # only HF_TOKEN is forwarded; the other keys in .env stay on the host
  if [ -f .env ]; then HF_TOKEN="$(set -a; . ./.env >/dev/null 2>&1; printf %s "$HF_TOKEN")"; export HF_TOKEN; fi
  # the proxy re-signs TLS with a corporate CA; reuse the host trust store (includes it)
  hf_flags=(-e HF_TOKEN -v /etc/ssl/certs/ca-certificates.crt:/etc/ssl/certs/host-ca.crt:ro
            -e SSL_CERT_FILE=/etc/ssl/certs/host-ca.crt -e REQUESTS_CA_BUNDLE=/etc/ssl/certs/host-ca.crt
            -e CURL_CA_BUNDLE=/etc/ssl/certs/host-ca.crt)
fi
if [ "$variant" = polar ]; then
  image=null-adapter:polar
  simcmf="$(cd "${SIMCMF_ROOT:-../ext/SimCMF}" && pwd)"
  hf_flags=(-v "$simcmf":/simcmf:ro -e SIMCMF_ROOT=/simcmf -e PY=python)
fi
if [ "$variant" != cpu ] && [ "${GPUS:-all}" != none ]; then gpu_flags=(--gpus "${GPUS:-all}"); fi
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY \
  --ipc=host "${gpu_flags[@]}" "${hf_flags[@]}" -v "$PWD":/work -w /work \
  --cpus="${CPUS:-16}" "$image" "$@"
