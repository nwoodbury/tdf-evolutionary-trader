"""
From the ME 575 notes: chapter 5 examples 5-7, used for testing
multi-objective, continuous optimization.
"""

from genetic import GA as ga


if __name__ == '__main__':
    variables = ['x1', 'x2']
    definition = {
        'x1': {
            'lb': -100,
            'ub': 100
        },
        'x2': {
            'lb': -100,
            'ub': 100
        }
    }
    objective = [
        []
    ]
