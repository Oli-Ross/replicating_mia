import yaml
import datetime
import os.path
import os

configDir = os.path.join(os.path.dirname(__file__), "..", "config")
if not os.path.isdir(configDir):
    os.mkdir(configDir)


class ConfigFilePath:
    """
    Contains the absolute path to a configuration file.

    It provides class methods to construct this path from a relative path, the
    current time or simply a name (the last 2 are useful if the current
    configuration should be saved to disk and assume that `config/` is used).
    """

    def __init__(self, absoluteFilePath: str) -> None:
        self.absoluteFilePath = absoluteFilePath

    @classmethod
    def from_rel_path(cls, relativeFilePath: str):
        """
        Construct ConfigFile object from the relative file path.
        """
        curDir = os.path.abspath(os.path.curdir)
        absoluteFilePath = os.path.join(curDir, relativeFilePath)
        return cls(absoluteFilePath)

    @classmethod
    def from_name(cls, name: str):
        """
        Construct ConfigFile object from a chosen file name, which is assumed to
        be located in `config/`.
        """
        absoluteFilePath = os.path.join(configDir, name)
        return cls(absoluteFilePath)

    @classmethod
    def from_current_time(cls):
        """
        Construct ConfigFile object using the current time, and assume it's
        located in `config/`.
        """
        now = datetime.datetime.now().strftime("%d.%m.%Y_%H:%M")
        fileName = f"{now}.yml"
        configFilePath = os.path.join(configDir, fileName)
        return cls(configFilePath)


class Configuration:
    """
    Holds configuration data.

    Provides class methods to parse a YAML file into a Configuration object or
    save a Configuration object to disk.
    This happens dynamically, the attributes depend on the YAML file's content.
    """

    def __init__(self) -> None:
        pass

    def save_to_file(self, configName: str | None = None):
        """
        Save the Configuration object's content into a yaml file.

        If a `configName` is passed, this will be used to construct the file
        name. Otherwise the current time is used. Either way, config files are
        written into the `config/` directory.
        """
        if configName is None:
            configFilePath = ConfigFilePath.from_current_time().absoluteFilePath
        else:
            configFilePath = ConfigFilePath.from_name(
                configName).absoluteFilePath
        with open(configFilePath, 'w') as file:
            yaml.dump(self, file)
        print(f"Configuration has been saved to {configFilePath}.")

    @classmethod
    def from_file(cls, configFilePath: ConfigFilePath):
        """
        Return a Configuration object loaded from a given file path.
        """
        absoluteFilePath = configFilePath.absoluteFilePath
        assert os.path.isfile(absoluteFilePath)
        with open(absoluteFilePath) as yamlFile:
            return yaml.load(yamlFile, Loader=yaml.Loader)

    @classmethod
    def from_rel_path(cls, fileName: str):
        """
        Conveniance method to load a Configuration from relative file path.
        """
        configFilePath = ConfigFilePath.from_rel_path(fileName)
        return cls.from_file(configFilePath)

    @classmethod
    def from_name(cls, fileName: str):
        """
        Conveniance method to load a Configuration from `config/`.

        `fileName` is the name of the file inside `config/`.
        """
        configFilePath = ConfigFilePath.from_name(fileName)
        return cls.from_file(configFilePath)
