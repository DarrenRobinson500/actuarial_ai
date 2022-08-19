import pandas as pd
import numpy as np
import tensorflow as tf
# import math
# from keras.datasets import boston_housing
from keras import layers, Sequential
from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
from keras import Input, Model
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


from constants import *
# Read the file
# dtypes = {"age": "category", "adviser": "category",}
# data = pd.read_csv('rate.csv', dtype=dtypes, usecols=list(dtypes) + ["lapse"],).to_numpy()
data = pd.read_csv('rate.csv').to_numpy()
# data = data.to_numpy()
# print(data)

rates = data[:,2].reshape(15,1)
data = data[:,:2]
print(data)

# rates -= rates.mean()
# rates /= rates.std()
# print("Pre", type(rates))
# rates = np.asarray(rates).astype('float32')
# print("Post", type(rates))
# print(rates)

print("Train data:", data.shape)
print("Train target:", rates.shape)

print(rates)

# Encode the data
age = Input(shape=(1,), name="age")
adviser = Input(shape=(1,), name="adviser", dtype="int64")

age_encoded = encode_numerical_feature(age, "age", train_ds)
adviser_encoded = encode_categorical_feature(adviser, "adviser", train_ds, False)

all_inputs = [age, adviser,]
all_features = concatenate(
    [
        age_encoded,
        adviser_encoded,
    ]
)

# Create and run model
x = Dense(32, activation="relu")(all_features)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=50, validation_data=val_ds)





# 2. MODEL
layer1 = layers.Dense(2, activation="relu")
layer2 = layers.Dense(16, activation="relu")
layer3 = layers.Dense(16, activation="relu")
layer4 = layers.Dense(1)
model = Sequential([layer1, layer2, layer3, layer4])
model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["mae",],
)

# [print(i.shape, i.dtype) for i in model.inputs]
# [print(o.shape, o.dtype) for o in model.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in model.layers]

# Run
history = model.fit(data, rates, epochs=20, batch_size=512,)

# val_mse, val_mae = model.evaluate(data, rates, verbose=0)
def create_lapse_rate_model(adviser):
    lapse_rate_actual = []
    for age in AGE_RANGE:
        sample = {"age": age, "adviser": adviser, }
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        print(input_dict)
        prediction = model.predict(input_dict)[0][0]
        print("Create lapse rate model", prediction)
        lapse_rate_actual.append(prediction)
    print(lapse_rate_actual)
    return lapse_rate_actual

def create_lapse_rate_actual(adviser):
    lapse_rate_actual = []
    for age in AGE_RANGE:
        lapse_rate_actual.append(lapse_rate_calc(age, adviser))
    return lapse_rate_actual

def plot_data():
    for x in ADVISER_RANGE:
        lapse_rates_model = create_lapse_rate_model(x)
        lapse_rates_actual = create_lapse_rate_actual(x)
        label_m = f"Adviser {x} (modelled)"
        label_a = f"Adviser {x} (actual)"
        plt.plot(AGE_RANGE, lapse_rates_model, label=label_m)
        plt.plot(AGE_RANGE, lapse_rates_actual, label=label_a)
    plt.title("Lapse rates")
    plt.xlabel("Age")
    plt.ylabel("Lapse rate")
    plt.legend()
    plt.show()

plot_data()