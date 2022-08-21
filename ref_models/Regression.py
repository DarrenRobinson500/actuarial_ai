import numpy as np
from keras.datasets import boston_housing
from keras import layers, Sequential

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# 1. DATA
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data -= mean
train_data /= std
test_data -= mean
test_data /= std

print("Train data:", train_data.shape)
print("Train target:", train_targets.shape)

# 2. MODEL
def build_model():
    layer1 = layers.Dense(64, activation="relu")
    layer2 = layers.Dense(64, activation="relu")
    layer_final = layers.Dense(1)
    model = Sequential([layer1, layer2, layer_final])

    model.compile(
        optimizer="rmsprop",
        loss="mse",
        metrics=["mae"],
    )
    return model

# 3. RUN
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    start = i * num_val_samples
    end = (i + 1) * num_val_samples
    val_data = train_data[start:end]
    val_targets = train_targets[start:end]
    partial_train_data = np.concatenate([train_data[:start], train_data[end:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:start], train_targets[end:]], axis=0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# 4. PRESENT
print(all_scores)
print(np.mean(all_scores))