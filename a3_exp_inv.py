import pandas as pd
from constants import *


# Read the file
dtypes = {"age": "category", "adviser": "category",}
df = pd.read_csv('files/lapse.csv', dtype=dtypes, usecols=list(dtypes) + ["fum_s", "dur_s", "lapse"],)

# Group the data
grouping = ["age", "adviser"]
grouped = df.groupby(grouping)
print(grouped)

# Analysis
lapses_total = grouped['lapse'].sum()
policies_total = grouped['lapse'].count()
rate = lapses_total / policies_total

# Output
print(rate)

print(fn_lapse_rates)
rate.to_csv("files/rate.csv", index=True, header=True)
