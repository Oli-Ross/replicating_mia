from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import download
import datasets
import app_parse as parse
import app_setup as setup


def main():

    config = parse.parse_config()
    setup.set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = datasets.load_dataset(config["targetDataset"]["name"])
    targetModel = setup.set_up_target_model(config, targetDataset)
    shadowDataset = setup.get_shadow_data(config, targetDataset, targetModel)


if __name__ == "__main__":
    main()
