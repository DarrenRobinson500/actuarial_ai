from keras.layers import IntegerLookup, Normalization, StringLookup, Dense, Dropout, concatenate
import tensorflow as tf

AGE_RANGE = [20, 30, 40, 50, 60]
AGE_RANGE_NEW = [20, 30]
ADVISER_RANGE = [1, 2, 3]

fn_data_s = "files/data_s.csv"
fn_data_e = "files/data_e.csv"
fn_data_c = "files/data_c.csv"
fn_lapse_rates = "files/rate.csv"
fn_lapse_table = 'files/lapse_table.csv'
fn_output = "files/output.csv"

def lapse_rate_calc(age, adviser):
    lapse_rate = age / 100
    if adviser == 3 and age > 45:
        lapse_rate *= 3
    lapse_rate = min(lapse_rate, 1)
    return lapse_rate

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
