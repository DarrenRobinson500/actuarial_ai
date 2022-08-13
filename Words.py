# Binary Text classification

import numpy as np
import tensorflow as tf
import math
from keras.datasets import imdb
from keras import layers, Sequential
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()
def print_review(x):
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = " ". join([reverse_word_index.get(i-3, "?") for i in train_data[x]])
    label = train_labels[x]
    if label == 1: label = "Positive"
    else:          label = "Negative"
    print(f"{label}: {decoded_review}")

for x in range(5): print_review(x)

def vectorize_samples(samples, dimension=10000):
    results = np.zeros((len(samples), dimension))
    for i, sample in enumerate(samples):
        for j in sample:
            results[i,j] = 1
    return results

# DATA
x_train = vectorize_samples(train_data)
x_test = vectorize_samples(test_data)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print("Train data:", train_data.shape)
print("Test data:", test_data.shape)
print("Validation data:", x_val.shape)
print("Remaining Train data:", partial_x_train.shape)

# Model
layer1 = layers.Dense(8, activation="relu")
layer2 = layers.Dense(16, activation="relu")
layer3 = layers.Dense(16, activation="relu")
layer4 = layers.Dense(1, activation="sigmoid")
model = Sequential([layer1, layer4])
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy",],
)

# Run
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# Present results
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "bo", label="Training loss")
# plt.plot(epochs, val_loss_values, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
