"""
Decides the optimal portfolio for the CS 412 sample data.
"""

import pandas as pd

if __name__ == '__main__':
    histories = pd.read_csv('testdata.csv')
    variables = list(histories.axes[1])
    definition_inner = {
        'lb': 0,
        'ub': 100000,
        'category': 'discreet'
    }
    definition = {key: definition_inner for key in variables}
    curr_portfolio = {'uninvested_cash': 0}
    print definition
