import argparse
from typing import Dict
from os.path import isabs
import configuration as con


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Launch a membership inference attack pipeline')
    parser.add_argument(
        '--config',
        help='Relative path to config file.',
    )

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
