import numpy as np
import pandas as pd
import time
from b6_run_policy import *
start_time = time.time()

def analyse_change(policy):
    result0 = run_policy(data_s, policy).values[1]
    print("NPV:", result0)
    result1 = run_policy(data_e, policy).values[1]
    print("NPV:", result1)
    change = round(result1 - result0,2)
    print("dNPV:", change)


data_s = pd.read_csv('files/data_s.csv')
data_e = pd.read_csv('files/data_e.csv')
analyse_change(1)
analyse_change(9)

total_time = round(time.time() - start_time,2)
print(f"\n{total_time} seconds")
