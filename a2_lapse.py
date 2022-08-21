# 1. Create the hypothetical data file and export to csv

import pandas as pd

from constants import *

count_per_data_point = 10
train_samples = []
train_labels = []

def create_records(age, adviser, fum_s, dur_s, label, count):
    for _ in range(count):
        train_samples.append((age, adviser, fum_s, dur_s, label))
        train_labels.append(label)

def create_lapse_rate_actual(adviser):
    lapse_rate_actual = []
    for age in AGE_RANGE:
        lapse_rate_actual.append(lapse_rate_calc(age, adviser))
    return lapse_rate_actual

def create_data():
    fum_s = 10000
    dur_s = 5
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            lapse_rate = lapse_rate_calc(age, adviser)
            create_records(age, adviser, fum_s, dur_s, 1, int(count_per_data_point * lapse_rate))
            create_records(age, adviser, fum_s, dur_s, 0, int(count_per_data_point * (1-lapse_rate)))

# Create records and add them to a dataframe and dataset
create_data()
dataframe = pd.DataFrame(data=train_samples, index=None, columns=['age', 'adviser', 'fum_s', 'dur_s', 'lapse'])
print(dataframe)
dataframe.to_csv("files/lapse.csv", index=False, header=True)
