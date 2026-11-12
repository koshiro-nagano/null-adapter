#!/usr/bin/env bash
# Usage: ./build_docker.sh [cpu|gpu|phase2]   (default gpu; phase2 builds on the gpu image)
set -e
cd "$(dirname "$0")"
variant="${1:-gpu}"
file=Dockerfile; [ "$variant" = gpu ] && file=Dockerfile.gpu; [ "$variant" = phase2 ] && file=Dockerfile.phase2
docker build --network host -f "$file" \
  --build-arg http_proxy --build-arg https_proxy --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  -t "null-adapter:$([ "$variant" = phase2 ] && echo phase2 || echo "toy1-$variant")" .
