from os import environ
from os.path import isabs
import argparse
from typing import Dict, List

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import download
import configuration as con
import target_models as tm
import shadow_data as sd
import shadow_models as sm
import datasets as ds
import attack_data as ad


def set_seeds(seed: int):
    ds.set_seed(seed)
    tm.set_seed(seed)
    sd.set_seed(seed)
    #  attack_model.set_seed(seed)
    #  attack_data.set_seed(seed)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description='Launch a membership inference attack pipeline')
    parser.add_argument('--config', help='Relative path to config file.',)

    return vars(parser.parse_args())


def parse_config() -> Dict:
    options = parse_args()
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

    config = parse_config()
    set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = tm.get_target_model(config, targetDataset)
    shadowData = sd.get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = sd.split_shadow_data(config, shadowData)
    shadowModels, shadowDatasets = sm.get_shadow_models_and_datasets(config, shadowDatasets)
    attackDatasets = ad.get_attack_data(config, shadowModels, shadowDatasets)
    breakpoint()


if __name__ == "__main__":
    main()
