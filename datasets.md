This module provides an interface to the CIFAR-10, CIFAR-100 and Kaggle datasets.
Instantiate e.g. a Cifar-10 dataset like this:

```python
cifar10 = datasets.Cifar10Dataset()
```

Get a split for testing and training:

```python
(x_train, y_train), (x_test, y_test) = cifar10.split()
```

If you want the training subset to be sampled randomly, use the `random`
parameter. To set a fixed chosen size for the training subset, use the
`train_size` parameter.

```python
(x_train, y_train), (x_test, y_test) = cifar10.split(random=True,train_size=1000)
```

The Kaggle datasets has in-built functionality to cluster the data into fewer
classes by using the `KagglePurchaseDatasetClustered` class. To cluster the data
into 5 classes:

```python
kaggle_clustered_5 = datasets.KagglePurchaseDatasetClustered(5)
```
