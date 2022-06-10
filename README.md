# Replicating the membership inference attack

This repository contains my work on replicating the original paper of the 
[membership inference attack](https://arxiv.org/abs/1610.05820) 
against machine learning classifiers.

## Documentation

Documentation avaible at https://oliverross.github.io/replicating_mia.
Generate it locally with [pdoc](https://pypi.org/project/pdoc/):
```bash
cd mia && pdoc -o ../docs/build *.py
```

## Tests

Run tests with pytest:
```bash
cd mia && pytest -v -W ignore::DeprecationWarning
```

## Setup

Set up a virtual environment, e.g.: 
```bash
python -m venv .venv && source .venv/bin/activate
```
Install necessary packages: 
```bash
python -m pip install -r requirements.txt
```

A setup script is provided, which does:
* Download necessary datasets into `/data`
* Run tests
* Generate documentation into `/docs`

Invoke it with `python set_up.py`.
For more fine-grained control, see `python3 set_up.py -h`.
