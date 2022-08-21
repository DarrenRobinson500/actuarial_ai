# 1. Create the hypothetical data file and export to csv

import pandas as pd
from constants import *

count_per_data_point = 10

# Create a list for each file
data_s = []
data_e = []

def create_data():
    fum_s0 = 10000
    fum_s1 = 11000
    dur_s0 = 5
    dur_s1 = 6
    i = 1
    for age in AGE_RANGE:
        for adviser in ADVISER_RANGE:
            lapse_rate = lapse_rate_calc(age, adviser)
            for x in range(count_per_data_point):
                data_s.append((i, age, adviser, fum_s0, dur_s0))
                if x < int(count_per_data_point * (1-lapse_rate)):
                    data_e.append((i, age+1, adviser, fum_s1, dur_s1))
                i += 1

def list_to_csv(list, filename):
    df = pd.DataFrame(data=list, index=None, columns=['index', 'age', 'adviser', 'fum_s', 'dur_s'])
    df.to_csv("files/" + filename + ".csv", index=False, header=True)

# Create records and add them to a dataframe and dataset
create_data()
list_to_csv(data_s, "data_s")
list_to_csv(data_e, "data_e")

