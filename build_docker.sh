#!/usr/bin/env bash
# Usage: ./build_docker.sh [cpu|gpu|phase2|polar]   (default gpu; phase2 builds on the gpu image; polar is standalone)
set -e
cd "$(dirname "$0")"
variant="${1:-gpu}"
file=Dockerfile; [ "$variant" = gpu ] && file=Dockerfile.gpu; [ "$variant" = phase2 ] && file=Dockerfile.phase2
tag="toy1-$variant"; [ "$variant" = phase2 ] && tag=phase2
extra=()
if [ "$variant" = polar ]; then
  file=Dockerfile.polar; tag=polar
  # host trust store (contains the proxy's CA) for pip inside the build
  [ -f /etc/ssl/certs/ca-certificates.crt ] && extra=(--secret id=ca,src=/etc/ssl/certs/ca-certificates.crt)
fi
DOCKER_BUILDKIT=1 docker build --network host -f "$file" "${extra[@]}" \
  --build-arg http_proxy --build-arg https_proxy --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
  -t "null-adapter:$tag" .
