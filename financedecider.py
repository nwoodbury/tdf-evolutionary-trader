"""
Decides the optimal portfolio for the TDF data queried from April 7-8.
"""

import pandas as pd
import math
from genetic import GA as ga
import random


def expected_returns(x, variables, y, V, rbar, bids):
    """
    Computes one part of the objective: the expected returns.

    @param x {list of number} The current values of x_j for all j. Each entry
        in x corresponds to the variable in the same location in variables.
    @param variables {list of string} The list of symbols being considered
        for trade.
    @param y {list of number} The current values of y_j for all j. Each entry
        in y corresponds to the variable in the same location in variables.
    @param V {number} The current portfolio value.
    @param rbar {pandas.Series} The average expected returns of each security
        in variables.
    @param bids {pandas.Series} The most recent bid prices for each security
        in variables.

    @returns {number} The expected returns of the portfolio given x.
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
    Computes one part of the objective: the risk.

    @param x {list of number} The current values of x_j for all j. Each entry
        in x corresponds to the variable in the same location in variables.
    @param variables {list of string} The list of symbols being considered
        for trade.
    @param y {list of number} The current values of y_j for all j. Each entry
        in y corresponds to the variable in the same location in variables.
    @param V {number} The current portfolio value.
    @param r {pandas.DataFrame} The historical returns for each security in
        variables.
    @param rbar {pandas.Series} The average expected returns of each security
        in variables.
    @param bids {pandas.Series} The most recent bid prices for each security
        in variables.

    @returns {number} The expected risk of the portfolio given x.
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
    Computes one part of the objective: the transaction costs.

    @param x {list of number} The current values of x_j for all j. Each entry
        in x corresponds to the variable in the same location in variables.
    @param variables {list of string} The list of symbols being considered
        for trade.
    @param asks {pandas.Series} The most recent ask prices for each security
        in variables.
    @param bids {pandas.Series} The most recent bid prices for each security
        in variables.

    @returns {number} The calculated transaction costs given x.
    """
    j = 0
    expected = 0

    den = (asks + bids).sum()

    for symb in variables:
        tj = (asks[symb] - bids[symb]) / den
        expected += tj * math.fabs(x[j])
        j += 1
    return expected


def constraint(x, variables, y, V, asks, bids, alpha):
    """
    Computes the constraint that cash must be positive.

    @param x {list of number} The current values of x_j for all j. Each entry
        in x corresponds to the variable in the same location in variables.
    @param variables {list of string} The list of symbols being considered
        for trade.
    @param y {list of number} The current values of y_j for all j. Each entry
        in y corresponds to the variable in the same location in variables.
    @param V {number} The current portfolio value.
    @param asks {pandas.Series} The most recent ask prices for each security
        in variables.
    @param bids {pandas.Series} The most recent bid prices for each security
        in variables.
    @param alpha {number} A discount of V, either p or 2p from the paper
        depending on whether the constraint is the lower or upper bound
        on the cash.

    @returns {number} The value of the constraint given x.
    """
    j = 0
    const = -V * (1 - alpha)

    for (symb, val) in y.iteritems():
        if symb == 'uninvested_cash':
            continue
        const += val*bids[symb]

    for symb in variables:
        if x[j] >= 0:
            const += bids[symb] * x[j]
        else:
            const += asks[symb] * x[j]
        j += 1

    return const


def get_returns(data):
    """
    Transforms the data into a set of returns (row[k] - row[k-1]) / row[k-1].

    @param data {pandas.DataFrame} The data (either the historical ask prices
        or the historical bid prices; typically the historical bid prices)

    @returns {pandas.DataFrame} The computed returns.
    """
    curr = data[1:]
    prev = data[:-1]

    prev = prev.transpose()
    prev.columns = list(curr.axes[0])
    prev = prev.transpose()

    returns = (curr - prev) / prev
    rbar = returns.mean()

    return returns, rbar


def map_x_to_definition(x, definition):
    """
    Converts the list of x values into a dictionary mapping symbols from
    variables to the respective x value.

    @param x {list of number} The current values of x_j for all j. Each entry
        in x corresponds to the variable in the same location in definition.
    @param definition {dict} The definition of the variables where the keys
        are the variable symbols.
    """
    mapped = {}
    pos = 0
    for (symb, val) in definition.iteritems():
        mapped[symb] = x[pos]
        pos += 1
    return mapped


def runtrial(n, asks, bids, y, V, p, iscontrol, gensize=8):
    """
    Main execution, running a single experiment.

    @param n {int} The number of generations to run.
    @param asks {pandas.DataFrame} The historical ask prices.
    @param bids {pandas.DataFrame} The historical bid prices.
    @param y {dict or pandas.Series} The current portfolio composition.
    @param V {number} The portfolio value.
    @param p {number} The percentage of V that should be kept as cash (upper
        bound), where no more than 2p is kept in cash.
    @param iscontrol {boolean} True if the transaction costs constraint should
        be ingnored.
    @param gensize {int, default=8} The maximum generation size as well as the
        size of the starting generation.
    """
    returns, rbar = get_returns(bids)

    variables = list(asks.axes[1])
    definition_inner = {
        'lb': 0,
        'ub': 10000,
        'category': 'discreet'
    }
    definition = {key: definition_inner for key in variables}
    objective = [
        lambda x: (expected_returns(x, variables, y, V, rbar, bids.ix[-1]) -
                   expected_risk(x, variables, y, V, returns, rbar,
                                 bids.ix[-1])
                   ) * -1
    ]
    if not iscontrol:
        objective.append(lambda x: expected_transaction(x, variables,
                                                        asks.ix[-1],
                                                        bids.ix[-1]))
    ineq = [
        lambda x: constraint(x, variables, y, V, asks.ix[-1], bids.ix[-1], p),
        lambda x: -constraint(x, variables, y, V, asks.ix[-1], bids.ix[-1],
                              2*p),

    ]
    j = 0
    for symb in variables:
        ineq.append(lambda x: -(x[j] + y.get(symb, 0)))
        j += 1

    starting_gen = []
    for i in range(gensize - 2):
        curr_chromosome = []
        for j in range(len(variables)):
            curr_chromosome.append(random.randint(0, 2000))
        starting_gen.append(curr_chromosome)
    chr0 = []
    chr100 = []
    for i in range(len(variables)):
        chr0.append(0)
        chr100.append(100)
    starting_gen.append(chr0)
    starting_gen.append(chr100)

    problem = ga(name='Test', variables=variables, definition=definition,
                 objective=objective, starting_gen=starting_gen,
                 max_gen_size=gensize, total_generations=n, ineq=ineq,
                 beta=2, mutation_prob=0.15, crossover_prob=0.75)
    # print problem
    problem.solve()
    # print problem.current_generation_str()
    # print problem.solution_str()

    x = problem.solution['chromosome']
    # print x

    Er = expected_returns(x, variables, y, V, rbar, bids.ix[-1])
    # print 'Expected Returns = %.5f (should be near 0.00007)' % Er
    Ersk = expected_risk(x, variables, y, V, returns, rbar, bids.ix[-1])
    # print 'Expected Risk = %.5f (should be near 0.00126)' % Ersk
    g = constraint(x, variables, y, V, asks.ix[-1], bids.ix[-1], p)
    return pd.Series(map_x_to_definition(x, definition)), problem, Er, Ersk, g


def test_on_markowitz(asks, bids, markowitz):
    """
    Tests the objective and constraints using the portfolio suggested by
    the MAD model (markowitz is a misnomer). The results as well as their
    estimated approximated values are displayed. This is only used for
    validating the correctness of the objective and constraints as coded here.

    @param asks {pd.DataFrame} The historical ask prices.
    @param bids {pandas.DataFrame} The historical bid prices.
    @param markowitz {dict or pandas.Series} The portfolio selected by the
        MAD model.
    """
    print markowitz
    variables = list(asks.axes[1])
    curr_portfolio = {'uninvested_cash': 100000}
    portfolio_value = 100000
    returns, rbar = get_returns(bids)

    x = []
    for symb in variables:
        x.append(markowitz.loc[symb])
    print x
    Er = expected_returns(x, variables, curr_portfolio, portfolio_value,
                          rbar, bids.ix[-1])
    print 'Expected Returns = %.5f (should be near 0.00007)' % Er

    Ersk = expected_risk(x, variables, curr_portfolio, portfolio_value,
                         returns, rbar, bids.ix[-1])
    print 'Expected Risk = %.5f (should be near 0.00126)' % Ersk

    const = constraint(x, variables, curr_portfolio, portfolio_value,
                       asks.ix[-1], bids.ix[-1], 0.05)
    print 'Constraint = %.0f (should be <= 0, on the order of -5000)' % const
