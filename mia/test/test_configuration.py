import configuration as con


class TestConfiguration():
    def test_from_name(self):
        config = con.from_name("example.yml")

        assert config["seed"] == 1234
        assert config["name"] == "Example Configuration"
