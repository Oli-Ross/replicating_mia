import configuration


class TestConfiguration():
    def test_from_name(self):
        con = configuration.Configuration.from_name("example.yml")

        assert con.seed == 1234
        assert con.name == "Example Configuration"
