"""
From the ME 575 notes: chapter 5 examples 5-7, with integer variables
used for testing multi-objective, discreet optimization.
"""

from genetic import GA as ga


if __name__ == '__main__':
    variables = ['x1', 'x2']
    definition = {
        'x1': {
            'lb': -100,
            'ub': 100,
            'category': 'discreet'
        },
        'x2': {
            'lb': -100,
            'ub': 100,
            'category': 'discreet'
        }
    }
    objective = [
        lambda x: 10*x[0] - x[1],
        lambda x: (1 + x[1]) / x[0],
    ]
    starting_gen = [[1, 1],
                    [1, 8],
                    [7, 55],
                    [1, 0],
                    [3, 17],
                    [2, 11]]
    problem = ga(name='Example 5-7', variables=variables,
                 definition=definition, objective=objective,
                 starting_gen=starting_gen, trim_first=False,
                 total_generations=10000)
    print problem
    print problem.current_generation_str(True)

    problem.solve()
    print problem.current_generation_str()
    print problem.solution_str()
