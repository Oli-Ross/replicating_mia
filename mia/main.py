from os import environ
import argparse
from typing import Dict

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import download
import configuration as con
import target_models as tm
import shadow_data as sd
import shadow_models as sm
import datasets as ds
import attack_data as ad
import attack_model as am


def set_seeds(seed: int):
    ds.set_seed(seed)
    tm.set_seed(seed)
    sd.set_seed(seed)
    am.set_seed(seed)
    ad.set_seed(seed)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description='Launch a membership inference attack pipeline')
    parser.add_argument('--config', help='Relative path to config file.',)

    return vars(parser.parse_args())


def main():

    options = parse_args()
    config = con.from_cli_options(options)
    set_seeds(config["seed"])

    download.download_dataset(config["targetDataset"]["name"])

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = tm.get_target_model(config, targetDataset)
    shadowData = sd.get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = sd.split_shadow_data(config, shadowData)
    shadowModels, shadowDatasets = sm.get_shadow_models_and_datasets(config, shadowDatasets)
    attackDatasets = ad.get_attack_data(config, shadowModels, shadowDatasets)
    attackModels = am.get_attack_models(config, attackDatasets)
    overallAccuracy = am.evaluate_models(attackModels, attackDatasets)
    print(f"Average attack accuracy over all classes: {overallAccuracy}")


if __name__ == "__main__":
    main()
