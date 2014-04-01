"""
From the ME 575 notes: example 1, used for testing.
"""

from genetic import GA as ga


if __name__ == '__main__':
    variables = ['x1', 'x2']
    definition = {
        'x1': {
            'lb': 0,
            'ub': 0.5,
            'category': 'continuous'
        },
        'x2': {
            'lb': 0,
            'ub': 0.5,
            'category': 'continuous'
        }
    }
    objective = [
        [0, 1.429, 0.57]
    ]
    ineq = [
        [0, 2, 0],
        [0, 0, 2],
        [0.3386, 1.354, 1.323],
        [0.2463, 1.26, 1.232]
    ]
    eq = []
    starting_gen = [[0.2833, 0.1408],
                    [0.0248, 0.0316],
                    [0.1384, 0.4092],
                    [0.3229, 0.1386],
                    [0.0481, 0.1625],
                    [0.4921, 0.2845]]
    problem = ga('Two-bar Truss', variables, definition, objective, ineq, eq,
                 starting_gen, 6, False)
    print problem
    print problem.current_generation_str()

    selection_rnd = [0.5292, 0.0436, 0.2949, 0.0411, 0.9116, 0.7869,
                     0.3775, 0.8691, 0.1562, 0.5616, 0.8135, 0.4158,
                     0.7223, 0.3062, 0.1357, 0.5625, 0.2974, 0.6033]
    print problem.next_generation(2, selection_rnd)
