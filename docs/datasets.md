## Loading datasets

This module preprocesses the datasets (where necessary) and returns them as 
`tf.data.Dataset` objects.

The main interface is the function `load_dataset(datasetName:str)`. It loads a
`tf.data.Dataset` from disk and if none is found, creates one and saves it to
disk. That means, it can also be used in advance to ensure that the dataset is
stored to disk and can be quickly loaded during training.

Use it like this:
```python
kaggle = load_dataset("kaggle")
```

Valid `datasetName` values are: "cifar10", "cifar100", "kaggle", "kaggle_2",
"kaggle_10","kaggle_20","kaggle_50","kaggle_100". Using the key "kaggle_x" 
returns the Kaggle dataset, but clustered into `x` classes. If the dataset has 
to be constructed, a k-means clustering is done in the backend, so it can take 
a while.

## Shuffling

To shuffle a Kaggle dataset (e.g. to construct a training set from random 
samples) call `shuffle_kaggle`:
```python
kaggle = load_dataset("kaggle")
kaggle_shuffled = shuffle_kaggle(kaggle)
```

## Pre-loading for later use

There is a conveniance method `load_all_datasets`, which loads all datasets as
`tf.data.Dataset` and saves them to disk, so that they can be quickly loaded in
the future.

## Seeding

Use `set_seed` to set a global seed, used by all random functions:
```python
set_seed(1234)
```
