"""
Decides the optimal portfolio for the CS 412 sample data.
"""

import pandas as pd
import math
from genetic import GA as ga
import random


def expected_returns(x, variables, y, V, rbar, bids):
    """
    Given the current x, the current investment y, the total portofolio value
    V, and the average returns for each security r, computes the expected
    returns.
    """
    index = 0
    expected = 0
    for symb in variables:
        expected += (x[index] + y.get(symb, 0)) / float(V) * \
            bids[symb] * rbar[symb]
        index += 1
    return expected


def expected_risk(x, variables, y, V, r, rbar, bids):
    """
    Given the current x, the current investment y, the total portfolio value
    V, and the average returns for each security r, computes the expected
    risk.
    """
    expected = 0
    T = len(r.axes[0])

    for k in range(T):
        j = 0
        for symb in variables:
            expected += math.fabs((x[j] + y.get(symb, 0)) *
                                  bids[symb] / float(V) *
                                  (r.ix[k, j] - rbar[symb]))
            j += 1
    return expected / float(T)


def expected_transaction(x, variables, asks, bids):
    """
    Given the current x and current asks and bids, computes the expected
    transaction costs.
    """
    j = 0
    expected = 0

    den = (asks + bids).sum()

    for symb in variables:
        tj = (asks[symb] - bids[symb]) / den
        expected += tj * math.fabs(x[j])
        j += 1
    return expected


def constraint(x, variables, y, asks, bids, hc):
    j = 0
    const = y.get('uninvested_cash', 0) - hc

    for symb in variables:
        if x[j] >= 0:
            const -= bids[symb] * x[j]
        else:
            const -= asks[symb] * x[j]

        j += 1
    const *= -1
    return const


def runtrial(asks, bids):
    """
    Main execution.
    """
    asks = pd.read_csv('askdata.csv', parse_dates=True, index_col=0)
    bids = pd.read_csv('biddata.csv', parse_dates=True, index_col=0)

    curr = bids[1:]
    prev = bids[:-1]

    prev = prev.transpose()
    prev.columns = list(curr.axes[0])
    prev = prev.transpose()

    returns = 1000 * (curr - prev) / prev
    rbar = returns.mean()

    variables = list(asks.axes[1])
    definition_inner = {
        'lb': 0,
        'ub': 10000,
        'category': 'discreet'
    }
    definition = {key: definition_inner for key in variables}
    curr_portfolio = {'uninvested_cash': 100000}
    portfolio_value = 100000
    objective = [
        lambda x: (expected_returns(x, variables, curr_portfolio,
                                    portfolio_value, rbar, bids.ix[-1]) -
                   expected_risk(x, variables, curr_portfolio,
                                 portfolio_value, returns, rbar, bids.ix[-1]))
    ]
    ineq = [
        lambda x: constraint(x, variables, curr_portfolio, asks.ix[-1],
                             bids.ix[-1], 100000 * 0.5)
    ]
    j = 0
    for symb in variables:
        ineq.append(lambda x: -(x[j] + curr_portfolio.get(symb, 0)))
        j += 1

    starting_gen = []
    for i in range(10):
        curr_chromosome = []
        for j in range(len(asks.axes[1])):
            curr_chromosome.append(random.randint(0, 100))
        starting_gen.append(curr_chromosome)

    problem = ga(name='Test', variables=variables, definition=definition,
                 objective=objective, starting_gen=starting_gen,
                 total_generations=1000, ineq=ineq)
    print problem
    problem.solve()
    print problem.current_generation_str()
    print problem.solution_str()


if __name__ == '__main__':
    ASKS = pd.read_csv('askdata.csv', parse_dates=True, index_col=0)
    BIDS = pd.read_csv('biddata.csv', parse_dates=True, index_col=0)
    runtrial(ASKS, BIDS)
