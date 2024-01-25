from constants import *
import numpy as np
import pandas as pd
import math
import warnings
import time

start_time = time.time()

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

table = pd.read_csv(fn_lapse_table)

def run_policy_index(df, policy):
    policy_exists = (df['index'] == policy).any()
    if not policy_exists:
        result = pd.Series([0, 0])
        return result

    info = df.loc[df['index'] == policy]
    return run_policy(info)

def run_policy(info):
    # Data (for reading a series)
    age_s = info['age']
    fum_s = info['fum_s']
    adviser = info['adviser']
    dur_s = info['dur_s']

    if math.isnan(age_s): return pd.Series([0, 0])

    # Data (for reading a dataframe
    # age_s = info.iloc[0]['age']
    # fum_s = info.iloc[0]['fum_s']
    # adviser = info.iloc[0]['adviser']
    # dur_s = info.iloc[0]['dur_s']

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

    result = pd.Series([pv_fees, pv_profit])
    return result

def calc_lapse_rate(ages, adviser):
    result = []
    for age in ages:
        row = table.loc[(table['age'] == age) & (table['adviser'] == adviser)]
        rate = row.iloc[0]['lapse_rate']
        result.append(rate)
    result = pd.Series(result)
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

def run_policy_aoc(info):
    info_0 = info[1:5]
    info_1 = info[5:9]

    info_0.rename({"age_0": "age", "adviser_0": "adviser", "fum_s_0": "fum_s", "dur_s_0": "dur_s"}, inplace=True)
    output_0 = run_policy(info_0)
    output_0.rename({0: "fees_0", 1: "profit_0"}, inplace=True)

    info_1.rename({"age_1": "age", "adviser_1": "adviser", "fum_s_1": "fum_s", "dur_s_1": "dur_s"}, inplace=True)
    output_1 = run_policy(info_1)
    output_1.rename({0: "fees_1", 1: "profit_1"}, inplace=True)

    output = pd.concat([output_0, output_1])

    return output

def get_category(info):
    age_s_0 = info['age_0']
    age_s_1 = info['age_1']
    if math.isnan(age_s_0): return "new"
    if math.isnan(age_s_1): return "exit"
    return "continuing"

def run_all_aoc():
    # Get the data
    data = pd.read_csv(fn_data_c)

    # Categorise each row
    category = data.apply(get_category, axis=1)
    data_cat = pd.concat([data, category], axis=1)
    data_cat.rename(columns={0: "cat",}, inplace=True)

    # Add pv_profit and pv_fees
    output = data_cat.apply(run_policy_aoc, axis=1)
    data_cat_output = pd.concat([data_cat, output], axis=1)
    data_cat_output['profit_d'] = data_cat_output['profit_1'] - data_cat_output['profit_0']
    print("OUTPUT")
    print(data_cat_output)
    data_cat_output.to_csv(fn_output, index=False, header=True)

    # Group the data
    grouping = ["cat", "adviser_0", ]
    grouping = ["cat", ]
    grouped = data_cat_output.groupby(grouping)
    pd.options.display.float_format = '{:,.0f}'.format
    # value_0 = grouped['profit_0'].sum()
    # value_1 = grouped['profit_1'].sum()
    # count_b = grouped['profit_0', 'profit_1'].count()
    value_b = grouped['profit_0', 'profit_1', 'profit_d'].sum()
    # value_b.append(value_b.sum(numeric_only=True), ignore_index=True)
    value_b.loc['total'] = value_b.sum()

    # Print the summary
    print()
    print("SUMMARY")
    # print(count_b)
    print()
    print(value_b)

    # print(f'Start value: {sum(value_0):,.0f}')
    # print(f'End value: {sum(value_1):,.0f}')
    # print(f'Change in value: {sum(value_1 - value_0):,.0f}')



run_all_aoc()

total_time = round(time.time() - start_time, 3)
print(f"{total_time} seconds")
