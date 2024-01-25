import tensorflow as tf
import pandas as pd
from keras import Model
from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
from keras import Input
import matplotlib.pyplot as plt
import os.path
from a2_lapse import *

def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature

def create_lapse_rate_model(adviser):
    print()
    print("Create lapse rate model")
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

def create_data():
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            lapse_rate = lapse_rate_calc(age, adviser)
            create_records(age, adviser, 1, int(count_per_data_point * lapse_rate))
            create_records(age, adviser, 0, int(count_per_data_point * (1-lapse_rate)))

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("lapse")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

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

train_dataframe = pd.read_csv(fn_lapse_rates)
train_ds = dataframe_to_dataset(train_dataframe)

# Create batches
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)

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

# Define the  model
x = Dense(32, activation="relu")(all_features)
x = Dense(32, activation="relu")(x)
x = Dense(32, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)
model = Model(all_inputs, output)

# Compile and run the model
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, epochs=500)

# Plot the output vs expected
plot_data()

#Save the model
print(model.summary())
filename = "models/lapse.tf"
if os.path.isfile(filename) is False:
    model.save(filename)