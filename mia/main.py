from os import environ
import argparse
from typing import Dict

from invoke import run

import configuration as con
import attack_model as am
import attack_data as ad

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description='Launch a membership inference attack pipeline')
    parser.add_argument('--config', help='Relative path to config file.',)
    parser.add_argument(
        '--prepare',
        action="store_true",
        help="Prepare all data and train models to launch the attack.")
    parser.add_argument(
        '--evaluate',
        action="store_true",
        help="Evaluate attack accuracy by loading models and data from disk.")
    return vars(parser.parse_args())


def run_script(scriptName: str, configFile: str):
    run(f"python {scriptName} --config {configFile}", pty=True)


def prepare(configFile: str):

    run_script(f"download.py", configFile)
    run_script(f"datasets.py", configFile)
    run_script(f"target_models.py", configFile)
    run_script(f"shadow_data.py", configFile)
    run_script(f"shadow_models.py", configFile)
    run_script(f"attack_data.py", configFile)
    run_script(f"attack_model.py", configFile)
    run_script(f"attack_pipeline.py", configFile)


def evaluate(config: Dict):
    attackDatasets = ad.load_attack_data(config)
    attackModels = am.get_attack_models(config, attackDatasets)
    overallAccuracy = am.evaluate_models(attackModels, attackDatasets)
    print(f"Average attack accuracy over all classes: {overallAccuracy}")


if __name__ == "__main__":
    options = parse_args()
    if options["prepare"]:
        configFileName = options["config"]
        prepare(configFileName)

    if options["evaluate"]:
        config = con.from_cli_options(options)
        evaluate(config)
