import pandas as pd

import numpy as np
# import tensorflow as tf
import math
# from keras.datasets import boston_housing
# from keras import layers, Sequential
# import matplotlib.pyplot as plt

# Read the file
dtypes = {"age": "category", "adviser": "category",}
data = pd.read_csv('rate.csv', dtype=dtypes, usecols=list(dtypes) + ["lapse"],).to_numpy()
# data = data.to_numpy()
# print(data)

rates = data[:,2]

rates -= rates.mean()
rates /= rates.std()

# print(rates)

print(np.shape(data))
print(np.shape(rates))

rates = np.hstack((data, rates))
# print(rates)