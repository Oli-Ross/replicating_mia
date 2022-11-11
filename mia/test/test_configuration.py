import configuration as con
import pytest
import os.path
import os


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

    @pytest.mark.skip("Hardcoded path.")
    def test_from_abs_path(self):
        configFile = "/home/oliver/Oliver/Studium/Master_Informatik/master/code/config/example.yml"
        config = con.from_rel_path(configFile)

        assert config["seed"] == 1234
        assert config["name"] == "Example Configuration"

    def test_save_to_file(self):
        config = con.from_name("example.yml")
        con.save_to_file(config, "test.yml")

        curDir = os.path.dirname(__file__)
        configFile = os.path.join(curDir, "../../config/test.yml")

        try:
            assert os.path.isfile(configFile)
        except BaseException:
            pass
        finally:
            os.remove(configFile)
