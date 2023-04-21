from os import environ
from os.path import isabs
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


def parse_config(options: Dict) -> Dict:
    """
    Take options from CLI and load correct config file.
    """
    configFile = options["config"]
    try:
        if isabs(configFile):
            config = con.from_abs_path(configFile)
        else:
            config = con.from_rel_path(configFile)
        name = config["name"]
        print(f"Using configuration \"{name}\"")
    except BaseException:
        config = con.from_name("example.yml")
        print("Using default configuration.")
    return config


def main():

    options = parse_args()
    config = parse_config(options)
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
