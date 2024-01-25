import tensorflow as tf
from keras.models import load_model
import pandas as pd
from constants import *

filename = "models/lapse.tf"
model = load_model(filename)

AGE_RANGE = range(20,66)
ADVISER_RANGE = [1, 2, 3]

table = []
for age in AGE_RANGE:
    for adviser in ADVISER_RANGE:
        sample = {"age": age, "adviser": adviser, }
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        prediction = round(model.predict(input_dict)[0][0], 3)
        row = (age, adviser, prediction)
        table.append(row)
print(table)
dataframe = pd.DataFrame(data=table, index=None, columns=['age', 'adviser', 'lapse_rate'])
print(dataframe)
dataframe.to_csv(fn_lapse_table, index=False, header=True)

