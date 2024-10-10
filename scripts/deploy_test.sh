#!/bin/bash

set -euo pipefail

cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.."

poetry update
poetry run pip freeze > ./requirements.txt
podman build -t jane-the-llama .
podman compose restart || docker compose up
