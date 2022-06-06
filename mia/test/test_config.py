import config
import os.path


class TestConfigFilePath():
    def test_from_rel_path(self):
        filePath = "../config/test.yaml"
        con = config.ConfigFilePath.from_rel_path(filePath)
        actualPath = os.path.abspath(con.absoluteFilePath)
        expectedPath = os.path.abspath(
            os.path.join(
                config.configDir,
                "test.yaml"))

        assert expectedPath == actualPath


class TestConfiguration():
    def test_from_name(self):
        con = config.Configuration.from_name("example.yaml")

        assert con.seed["numpy"] == 1234
        assert con.name == "Example Configuration"
