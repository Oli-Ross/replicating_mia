import configuration as con
import os.path


class TestConfiguration():
    def test_from_name(self):
        config = con.from_name("example.yml")

        assert config["seed"] == 1234
        assert config["name"] == "Example Configuration"

    def test_from_rel_path(self):
        curDir = os.path.dirname(__file__)
        configFile = os.path.join(curDir, "../../config/example.yml")
        config = con.from_rel_path(configFile)

        assert config["seed"] == 1234
        assert config["name"] == "Example Configuration"
