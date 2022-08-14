import pandas as pd

# Read the file without grouping
# df = pd.read_csv('data.csv')
# print(df.to_string())

# Read the file with grouping
dtypes = {
    "age": "category",
    "adviser": "category",
}
df = pd.read_csv(
    'data.csv',
    dtype=dtypes,
    usecols=list(dtypes) + ["target"],
)
# print(df.to_string())
# grouped = df.groupby("age", "adviser")["adviser"].count()
# print()
print("WITH COUNT")
grouping = ["age", "adviser"]
grouped = df.groupby(grouping)
# print(grouped["target"].sum())

print("ITERATIONS")
for x, y in grouped:
    print(x)

lapses_total = grouped['target'].sum()
policies_total = grouped['target'].count()
rate = lapses_total / policies_total
print("RATE")
print(rate)


    # policies = data.count(axis=0)
    # # print(data)
    # print("LAPSES")
    # print(lapses)
    # print("POLICIES")
    # print(policies)
    # print("RATE")
    # rate = round(lapses/policies,2)
    # print(rate)

# print(df.groupby(grouping)["target"].count())

# by_state.get_group("PA")
