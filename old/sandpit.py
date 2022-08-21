import pandas as pd
import numpy as np


def calc_lapse_rate(ages, adviser):
    print("Calc lapse rate - age", ages)
    print("Calc lapse rate - adviser", adviser)
    result = []
    for age in ages:
        row = table.loc[(table['age'] == age) & (table['adviser'] == adviser)]
        rate = row.iloc[0]['lapse_rate']
        result.append(rate)
    result = pd.Series(result)
    return result


age_s = 30
time = np.arange(65 - age_s)
ages = age_s + time

table = pd.read_csv('../files/lapse_table.csv')

lapse_rate_series = calc_lapse_rate(ages, 1)
print(lapse_rate_series)


