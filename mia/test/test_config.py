import configuration
import os.path


class TestConfigFilePath():
    def test_from_rel_path(self):
        filePath = "../config/test.yml"
        con = configuration.ConfigFilePath.from_rel_path(filePath)
        actualPath = os.path.abspath(con.absoluteFilePath)
        expectedPath = os.path.abspath(
            os.path.join(
                configuration.configDir,
                "test.yml"))

        assert expectedPath == actualPath


class TestConfiguration():
    def test_from_name(self):
        con = configuration.Configuration.from_name("example.yml")

        assert con.seed == 1234
        assert con.name == "Example Configuration"
