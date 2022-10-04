This module defines the victim models and provides methods to save, load and
train them.

To set the seed:
```python
victim_models.set_seed(1234)
```

To load the models:
```python
cifar10Model = victim_models.CifarModel()
kaggleModel = victim_models.KaggleModel()
```
To train it with a `tf.data.Dataset` called `cifar10` (dimension has to fit):
```python
victim_models.train_model(cifar10Model, cifar10, epochs=1)
```

To save and load the model:
```python
victim_models.save_model("filename",cifar10Model)
cifar10Model = victim_models.load_model("filename")
```
