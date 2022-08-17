"""
.. include:: ../docs/configuration.md
"""

import datetime
import os
import os.path
from typing import Dict, Optional

import yaml

configDir = os.path.join(os.path.dirname(__file__), "..", "config")
if not os.path.isdir(configDir):
    os.mkdir(configDir)


class Configuration:
    """
    Holds configuration data.

    Provides methods to parse a YAML file into a Configuration object or
    save a Configuration object to disk.
    """

    def __init__(self, configOptions: Optional[Dict] = None) -> None:
        if configOptions:
            self.seed = configOptions["seed"]
            self.name = configOptions["name"]

    def save_to_file(self, configName: Optional[str] = None):
        """
        Save the Configuration object's content into a yaml file.

        If a `configName` is passed, this will be used to construct the file
        name. Otherwise the current time is used. Either way, config files are
        written into the `config/` directory.
        """
        if configName is None:
            now = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
            configName = f"{now}.yml"
        configFilePath = os.path.join(configDir, configName)
        with open(configFilePath, 'w') as file:
            yaml.dump(self, file)
        print(f"Configuration has been saved to {configFilePath}.")

    @classmethod
    def from_abs_path(cls, absoluteFilePath: str) -> 'Configuration':
        """
        Load a Configuration from absolute file path.
        """
        assert os.path.isfile(absoluteFilePath)
        with open(absoluteFilePath) as yamlFile:
            yamlContent: Dict = yaml.load(yamlFile, Loader=yaml.Loader)
        return cls(yamlContent)

    @classmethod
    def from_rel_path(cls, fileName: str) -> 'Configuration':
        """
        Load a Configuration from relative file path.
        """
        curDir = os.path.abspath(os.path.curdir)
        absoluteFilePath = os.path.join(curDir, fileName)
        return cls.from_abs_path(absoluteFilePath)

    @classmethod
    def from_name(cls, fileName: str) -> 'Configuration':
        """
        Load a Configuration from `config/`.

        `fileName` is the name of the file inside `config/`.
        """
        absoluteFilePath = os.path.join(configDir, fileName)
        return cls.from_abs_path(absoluteFilePath)
