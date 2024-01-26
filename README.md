# Replicating the membership inference attack

This repository contains my work on replicating the original paper of the 
[membership inference attack](https://arxiv.org/abs/1610.05820) 
against machine learning classifiers.

## Setup

It is recommended to use the docker setup to ensure the Cuda and cuDNN versions match the pinned tensorflow version.
The dockerfile uses `Cuda==11.2.1` and `cuDNN==8.1`, this project uses `tensorflow-gpu==2.9.3`.

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

The code is split into submodules in the subfolder `mia/`:
1. `download.py`: Download datasets
2. `datasets.py`: Data preprocessing
3. `target_models.py`: Train/load target model
4. `shadow_data.py`: Generate shadow data
5. `shadow_models.py`: Train/load shadow models
6. `attack_data.py`: Predict shadow data on shadow models and aggregate it into attack data
7. `attack_model.py`: Train/load attack model
8. `attack_pipeline.py`: Run the attack pipeline on the target model
9. (`configuration.py`: Parse config YAML file)

Each module can be called as a standalone script with the option `--config FILE` to read the configuration from `FILE`
(default is using `config/example.yml`).
By default datasets and models will be saved to disk and only generated or trained if they can't be loaded from disk.

`mia/main.py` ties the modules together to train/generate all models/data:

```bash
python mia/main.py --prepare
```

Subsequently, the attack model can be evaluated to determine the overall accuracy of the attack:
```bash
python mia/main.py --evaluate
```

Refer to the documentation of each module on how to use it as standalone. 

## Documentation

Documentation is avaible at https://oli-ross.github.io/replicating_mia.
You can generate it locally with [pdoc](https://pypi.org/project/pdoc/):
```bash
cd mia && pdoc -o ../docs/build *.py
```

## Tests

Run tests with pytest:
```bash
cd mia && pytest
```
