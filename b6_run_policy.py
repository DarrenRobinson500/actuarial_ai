from constants import *
import numpy as np
import pandas as pd

table = pd.read_csv('files/lapse_table.csv')

def run_policy(df, policy):
    policy_exists = (df['index'] == policy).any()
    if not policy_exists:
        result = pd.Series([0, 0])
        return result

    info = df.loc[df['index'] == policy]

    # Data
    age_s = info.iloc[0]['age']
    fum_s = info.iloc[0]['fum_s']
    adviser = info.iloc[0]['adviser']
    dur_s = info.iloc[0]['dur_s']

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
    # for x in variables_to_print:
    #     print(x)
    # return pv_profit

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
    data = pd.read_csv('data.csv')
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
