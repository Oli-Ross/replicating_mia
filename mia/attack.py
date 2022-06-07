import argparse
from typing import Dict

import datasets
from configuration import Configuration


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Launch a membership inference attack')
    parser.add_argument(
        '--config',
        help='Relative path to config file.',
    )

    return vars(parser.parse_args())


def set_seeds(config: Configuration):
    datasets.set_seed(config.seed)


if __name__ == "__main__":
    options = parse_args()
    try:
        config = Configuration.from_rel_path(options["config"])
        print("Using provided configuration.")
    except BaseException:
        config = Configuration.from_name("example.yml")
        print("Using default configuration.")

    set_seeds(config)
