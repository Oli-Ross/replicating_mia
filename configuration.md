This module allows the parsing of YAML config files into python objects.

To load a YAML file located in `config/` use the method `from_name`:

```python
config = configuration.Configuration.from_name("example.yml")
```

Access the information of the YAML file by accessing members of the returned
object. E.g. if the YAML file has an entry "seed: 1234":

```python
print(config.seed)
1234
```

You can also load from a relative or absolute file path:

```python
config_relative = configuration.Configuration.from_rel_path("../example.yml")
config_absolute = configuration.Configuration.from_abs_path("/home/user/config/example.yml")
```
Save the current configuration to `config/` using the `save_to_file` method:

```python
config.save_to_file() # Uses current time as filename
config.save_to_file("example.yaml") 
```
