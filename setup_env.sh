#!/bin/bash
set -e

python3.9 -m venv --copies venv
source venv/bin/activate

pip install wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=neuroNoodle

pip install pre-commit
pre-commit install

deactivate
