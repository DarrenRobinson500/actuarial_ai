import pandas as pd
import time

start_time = time.time()

# Read the file
dtypes = {"age": "category", "adviser": "category",}
df = pd.read_csv('files/data.csv', dtype=dtypes, usecols=list(dtypes) + ["fum_s", "dur_s", "lapse"],)

# Group the data
grouping = ["age", "adviser"]
grouped = df.groupby(grouping)

# Analysis
lapses_total = grouped['lapse'].sum()
policies_total = grouped['lapse'].count()
rate = lapses_total / policies_total

# Output
print(rate)

rate.to_csv("files/rate.csv", index=True, header=True)

total_time = round(time.time() - start_time, 3)
print(f"{total_time} seconds")

