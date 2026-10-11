#!/bin/bash
set -e

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

pip install -e . --no-deps
pip install --upgrade math-verify nltk emoji syllapy langdetect wandb==0.27.2

apt-get update && apt-get install -y firejail

echo "ready"
