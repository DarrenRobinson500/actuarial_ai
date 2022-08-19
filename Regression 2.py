import numpy as np
from keras.datasets import boston_housing
from keras import layers, Sequential

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(type(train_data))

# 1. DATA
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

print("Train data:", train_data.shape)
print(train_data)
print()
print("Train target:", train_targets.shape)
print(train_targets)

# 2. MODEL
layer1 = layers.Dense(64, activation="relu")
layer2 = layers.Dense(64, activation="relu")
layer_final = layers.Dense(1)
model = Sequential([layer1, layer2, layer_final])

model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["mae"],
)

# 3. RUN
# model.fit(train_data, train_targets, epochs=5, batch_size=16, verbose=0)
# val_mse, val_mae = model.evaluate(test_data, test_targets, verbose=0)

# 4. PRESENT
# print(val_mse)
