import pandas as pd
import numpy as np


s = pd.Series([2, 3, 4])
print("PRE")
print(s)
s.rename({1: 4, 2: 5}, inplace=True)
print("POST")
print(s)