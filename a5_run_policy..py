import numpy as np
import pandas as pd
from keras.models import load_model
import time
from constants import *
start_time = time.time()

filename = "models/lapse.tf"
model = load_model(filename)

def calc_lapse_rate(age, adviser):
    print("Calc lapse rate - age", age)
    print("Calc lapse rate - adviser", adviser)
    result = []
    for x in age:
        sample = {"age": x, "adviser": adviser, }
        input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
        prediction = round(model.predict(input_dict)[0][0], 3)
        result.append(prediction)
    print(result)
    result = pd.Series(result)
    return result

def run_policy(info):
    print("Run policy")
    print(type(info))
    print(info)

    # Data
    age_s = info['age']
    fum_s = info['fum_s']
    adviser = info['adviser']
    dur_s = info['dur_s']

    # Time
    time = np.arange(65-age_s)
    age = age_s + time

    # Economic Assumptions
    discount_rate = 0.10
    inv_earnings = 0.07

    disc_factor = (1 / (1 + discount_rate)) ** time

    # Policy Assumptions
    lapse_rate = calc_lapse_rate(age, adviser)

    # Product Information
    fee_rate = 0.02
    expense_rate = 0.01

    # Calcs
    policies = (1 - lapse_rate) ** time
    fum_pp_factor = (1 + inv_earnings) ** time
    fum_pp = fum_s * fum_pp_factor
    fum = fum_pp * policies
    fees = fum * fee_rate
    expense = fum * expense_rate
    profit = fees - expense
    profit_disc = profit * disc_factor
    fees_disc = fees * disc_factor
    pv_fees = round(sum(fees_disc),2)
    pv_profit = round(sum(profit_disc),2)


    # variables_to_print = [time, dur, disc_factor_round, FUM_pp, fees, expense, profit, npv]
    variables_to_print = [time, lapse_rate]
    # variables_to_print = [npv]
    for x in variables_to_print:
        print(x)
    # return pv_profit

    result = pd.Series([pv_fees, pv_profit])
    return result

def run_all():
    data = pd.read_csv('files/data.csv')
    output = data.apply(run_policy, axis=1)
    data_output = pd.concat([data, output], axis=1)
    data_output.rename(columns={0: "fees", 1: "profit"}, inplace=True)
    print("OUTPUT")
    print(data_output)

    grouping = ["age", "adviser"]
    grouped = data_output.groupby(grouping)

    # Analysis
    pd.options.display.float_format = '{:,.0f}'.format

    value = grouped['profit'].sum()
    print()
    print("SUMMARY")
    print(value)
    print(type(value))

    print(f'Total Value: {sum(value):,.0f}')


# data = {"age": 20, "fum_s":10000, "dur_s": 3, "adviser": 1}
# result = run_policy(data)
# print(result)
# for age in [20,25,30]:
#     for adviser in [1,]:
#         rate = calc_lapse_rate(age, adviser)
#         print(age, adviser, rate)
run_all()

total_time = round(time.time() - start_time,2)
print(f"{total_time} seconds")
