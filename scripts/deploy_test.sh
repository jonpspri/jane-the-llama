#!/bin/bash

set -euo pipefail

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.."

poetry update
# poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt
poetry export --without-hashes -f requirements.txt -o requirements.txt
podman build -t jane-the-llama .
# podman compose restart || docker compose up
