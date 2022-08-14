import pandas as pd

# Read the file
dtypes = {"age": "category", "adviser": "category",}
df = pd.read_csv('data.csv', dtype=dtypes, usecols=list(dtypes) + ["target"],)

# Group the data
grouped = df.groupby(["age", "adviser"])

# Analysis
lapses_total = grouped['target'].sum()
policies_total = grouped['target'].count()
rate = lapses_total / policies_total

# Output
print(rate)
