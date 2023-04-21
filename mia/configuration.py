"""
.. include:: ../docs/configuration.md
"""

import datetime
import os
import os.path
from os.path import isabs
from typing import Dict, Optional

import yaml

configDir = os.path.join(os.path.dirname(__file__), "..", "config")
if not os.path.isdir(configDir):
    os.mkdir(configDir)


def save_to_file(configuration: Dict, configName: Optional[str] = None):
    """
    If a `configName` is passed, this will be used to construct the file
    name. Otherwise the current time is used. Either way, config files are
    written into the `config/` directory.
    """
    if configName is None:
        now = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
        configName = f"{now}.yml"
    configFilePath = os.path.join(configDir, configName)
    with open(configFilePath, 'w') as file:
        yaml.dump(configuration, file)
    print(f"Configuration has been saved to {configFilePath}.")


def from_abs_path(absoluteFilePath: str) -> Dict:
    """
    Load a Configuration from absolute file path.
    """
    assert os.path.isfile(
        absoluteFilePath), f"{absoluteFilePath} not found"
    with open(absoluteFilePath) as yamlFile:
        yamlContent: Dict = yaml.load(yamlFile, Loader=yaml.Loader)
    return yamlContent


def from_rel_path(fileName: str) -> Dict:
    """
    Load a Configuration from relative file path.
    """
    curDir = os.path.abspath(os.path.curdir)
    absoluteFilePath = os.path.join(curDir, fileName)
    return from_abs_path(absoluteFilePath)


def from_name(fileName: str) -> Dict:
    """
    Load a Configuration from `config/`.

    `fileName` is the name of the file inside `config/`.
    """
    absoluteFilePath = os.path.join(configDir, fileName)
    return from_abs_path(absoluteFilePath)


def from_cli_options(options: Dict) -> Dict:
    """
    Take options from CLI and load correct config file.
    """
    configFile = options["config"]
    try:
        if isabs(configFile):
            config = from_abs_path(configFile)
        else:
            config = from_rel_path(configFile)
        name = config["name"]
        print(f"Using configuration \"{name}\"")
    except BaseException:
        config = from_name("example.yml")
        print("Using default configuration.")

    return config
