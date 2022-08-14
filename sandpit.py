import numpy as np
from keras.datasets import boston_housing
from keras import layers, Sequential

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(type(train_data))
print(train_data)