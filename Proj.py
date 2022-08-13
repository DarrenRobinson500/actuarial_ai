import numpy as np

# Time
time = np.arange(10)

# Economic Assumptions
discount_rate = 0.10
inv_earnings = 0.07

disc_factor = (1 / (1 + discount_rate)) ** time
disc_factor_round = np.around(disc_factor, 4)

# Policy Assumptions
lapse_rate = 0.10
FUM_s = 10000

# Data
dur_s = 5
dur = time + dur_s

policies = (1 - lapse_rate) ** time

FUM_pp_factor = (1 + inv_earnings) ** time
FUM_pp = FUM_s * FUM_pp_factor

FUM = FUM_pp * policies

print(time)
print(dur)
print(disc_factor_round)
print(FUM_pp_factor)
print(FUM_pp)
print(policies)
print(FUM)
