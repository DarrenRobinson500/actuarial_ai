import pandas as pd

df0 = pd.read_csv('files/data_s.csv')
df1 = pd.read_csv('files/data_e.csv')

keys = list(df1.columns.values)
i0 = df0.set_index(keys).index
i1 = df1.set_index(keys).index

df0['lapse'] = ~i0.isin(i1)
print(df0)
df0.to_csv("files/lapse.csv", index=False, header=True)
