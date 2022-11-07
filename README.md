# Replicating the membership inference attack

This repository contains my work on replicating the original paper of the 
[membership inference attack](https://arxiv.org/abs/1610.05820) 
against machine learning classifiers.

## Setup

### Venv

Set up a virtual environment, e.g.: 
```bash
python -m venv .venv && source .venv/bin/activate
```
Install dependencies: 
```bash
python -m pip install -r requirements.txt
```

### Docker

Alternatively, use the provided dockerfile.
To leverage the GPU, install the respective NVIDIA driver on the host system, as
well as the [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-docker).
Build the image:
```bash
docker build -t mia:0.1 .
```

Set the gid and uid as environment variables, so that they are set correctly in
the container.
```bash
export UID=$(id -u)
export GID=$(id -g)
```

Start a container:
```bash
docker compose run mia
```

Verify that the GPU is available in `tensorflow`:
```bash
TF_CPP_MIN_LOG_LEVEL=2 python -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')"
```

## Usage

The code is split into submodules:
1. `mia/download`: Download datasets
2. `mia/datasets`: Data preprocessing
3. `mia/configuration`: Parse config YAML file
4. `mia/target_models`: Target model definition
5. `mia/attack_model`: Attack model definition
6. `mia/attack_data`: Generate shadow data from original data
7. `mia/shadow_data`: Generate shadow data using paper's methods

Refer to the documentation of each module on how to use it. 
An example script, which ties the modules together is given at `mia/main.py`.

## Documentation

Documentation is avaible at https://oliverross.github.io/replicating_mia.
You can generate it locally with [pdoc](https://pypi.org/project/pdoc/):
```bash
cd mia && pdoc -o ../docs/build *.py
```

## Tests

Run tests with pytest:
```bash
cd mia && pytest
```
