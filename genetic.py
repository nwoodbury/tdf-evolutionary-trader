"""
An implementation of a multi-objective value-based genetic algorithm.
"""

from termcolor import colored
import random
import math
from copy import deepcopy


def is_equal(x, y, tol=0.001):
    """
    Utility for assertions. Returns true if x == y within tol tolerance.

    @param x {number} The actual value.
    @param y {number} The expected value.
    @param tol {number, default = 0.0001} The test tolerance.

    @returns {boolean}
    """
    return x <= y + tol and x >= y - tol


class GA:

    def __init__(self, **kwargs):
        """
        Algorithm initialization.

        Note that all the following parameters are default kwargs.

        @param name {string} The name of the problem.
        @param variables {list of string} A list of variable names in the order
            in which they are defined in the constraints and genes.
        @param definition {dict=>dict} Each key is a name (which must be in
            variables, and everything in variables must be a key). The value
            is a definition of the variable, defined by a dict. The definition
            dictionary may contain the following keys and their respective
            value types:
                'ub': {number, default=Inf} The upper bound on the feasible
                    values for the variable
                'lb': {number, default=-Inf} The lower bound on the feasible
                    values for the variable
                'category': {string in {'continuous', 'discrete'},
                    default='continuous'}. Whether the variable is continuous
                    or discrete.
        @param objective {list} The definition for the objective, which is to
            minimize: objective[0] + objective[1]*variables[0] +
                      objective[2]*variables[1] - ...

        @param ineq {list} The definition for the inequality constraints where:
            ineq[0] - ineq[1]*variables[0] - ineq[2]*variables[1] - ... <= 0
        @param eq {list} The definition for the equality constraints where:
            eq[0] - eq[1]*variables[0] - eq[2]*variables[1] - ... = 0
        @param starting_gen {list} The list of the starting generation genes,
            where each entry is a list defining the values for the defined
            variables.
        @param max_gen_size {int, default=len(starting_gen)} The maximum
            allowed genes in a generation. This is ignored for multi-objective
            optimization problems.
        @param crossover_prob {number, default=0.6} The probability of
            performing a crossover.
        @param mutation_prob {number, default=0.1} The probability of
            performing a mutation.
        @param total_generations {int, default=10} The total number of
            generations to be found by the algorithm. Since the given starting
            generation is 1, the algorithm will iterate total_generations-1
            times.
        @param beta {number, default=5} The mutation parameter
        @param trim_first {boolean} True if the initial generation should
            be immediately trimmed and sorted (False is useful for testing)

        @throws {Exception} If there exists a variable in variables that is
            not defined in definitions.
        @throws {Exception} If there exists a defined variable in definitions
            that is not in variables.
        @throws {Exception} If the length of each objective is not one greater
            than the number of variables.
        @throws {Exception} If the length of each constraint is not one greater
            than the number of variables.
        @throws {Exception} If the length of each gene in the starting
            generation is not the same as variables.
        """
        # Set All Variables
        self.name = kwargs.get('name', 'UNAMED')
        self.variables = kwargs.get('variables', [])
        self.definition = self.__populate_definition(
            kwargs.get('definition', {}))
        self.objective = kwargs.get('objective', [])
        self.ineq = kwargs.get('ineq', [])
        self.eq = kwargs.get('eq', [])
        self.starting_gen = kwargs.get('starting_gen', [])
        self.max_gen_size = kwargs.get('max_gen_size', len(self.starting_gen))
        self.crossover_prob = kwargs.get('crossover_prob', 0.6)
        self.mutation_prob = kwargs.get('mutation_prob', 0.1)
        self.total_generations = kwargs.get('total_generations', 10)
        self.beta = kwargs.get('beta', 5)
        self.trim_first = kwargs.get('trim_first', True)

        # Check Feasibilities (whether given parameters are allowed)
        self.__check_variables_definition_feasibility()
        self.__check_objective_feasibility()
        self.__check_constraints_feasibility()
        self.__check_initial_conditions_feasibility()

        # Initialize current generation
        self.__initialize_current_generation()

    def __str__(self):
        """
        Returns a string representation of the problem (not its current
        status or solution, essentially the initialization inputs).

        @returns {string}
        """
        strng = '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        strng += 'Problem: %s\n\n' % self.name

        strng += 'Problem Definition (%i design variables):\n' % \
            (len(self.definition))
        for (var, defn) in self.definition.iteritems():
            strng += '\t%s %s\n' % (var, self.__definition2str(defn))

        strng += 'Objective(s) (%i total):\n' % len(self.objective)
        for obj in self.objective:
            strng += '\t%s\n' % self.__objective2str(obj)

        strng += 'Constraint (%i inequality and %i equality)\n' % \
            (len(self.ineq), len(self.eq))
        for const in self.ineq:
            strng += '\t%s\n' % self.__const2str(const, '<=')

        strng += 'Maximum Generation Size = %i\n' % self.max_gen_size
        strng += 'Starting Generation:\n'
        for chromosome in self.starting_gen:
            strng += '\t%s\n' % self.__chromosome2str(chromosome)

        strng += '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        return strng

    def current_generation_str(self, verbose=False):
        """
        Returns a string representation of the current generation.

        @param verbose {boolean} If true, shows all fi and gi.

        @returns {string}
        """
        strng = '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        strng += 'Current generation for Problem: %s\n\n' % self.name

        j = 1
        for chromosomerep in self.generation:
            infeasible_string = ''
            if not is_equal(chromosomerep['g'], 0, 0.0001):
                infeasible_string = colored(' (Infeasible)', 'red')
            strng += 'Design %i%s: %s\n' % (j, infeasible_string,
                                            self.__chromosome2str(
                                                chromosomerep['chromosome']))

            strng += colored('\tfitness = %.4f\n' % chromosomerep['fitness'],
                             'green')

            if verbose:
                for i in range(1, len(self.objective) + 1):
                    strng += '\tf%i = %.4f\n' % (i, chromosomerep['f%i' % i])
                for i in range(1, len(self.ineq) + len(self.eq) + 1):
                    strng += '\tg%i = %.4f\n' % (i, chromosomerep['g%i' % i])
                strng += '\tg = %.4f\n' % chromosomerep['g']

            j += 1

        strng += '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        return strng

    def next_generation(self, tournament_size=2,
                        selection_rand=None, mutation_rand=None,
                        verbose=False):
        """
        Performs selection, crossover, and mutation to select the next
        generation. We assume elitism, meaning if the parent is better than
        the child, the parent is kept.

        @param tournament_size {numberi, default=2} The size of the tournament
            for tournament selection.
        @param selection_rand {list, default=None} If None, selection will
            use a random number generator. Otherwise, random numbers will be
            taken in sequence from the given list.
        @param mutation_rand {list, default=None} If None, mutation will use
            a random number generator. Otherwise, random numbers will be
            taken in sequence from the given list.
        @param verbose {boolean, default=False} True if output lines should
            be printed for debugging.

        @returns {boolean} True if the next generation is different from the
            current generation.
        """
        self.generation_number += 1
        new_generation = self.__selection(tournament_size, selection_rand)
        if verbose:
            print 'New generation after selection:'
            print new_generation
        mutated_generation = self.__mutation(new_generation, mutation_rand)
        if verbose:
            print 'New generation after mutation:'
            print mutated_generation
        for chromosome in mutated_generation:
            self.__add_chromosome(chromosome)
        self.__trim_generation()

    # =========================================================================
    # Private Helpers: Selection, Crossover, and Mutation
    # =========================================================================

    def __selection(self, tournament_size, selection_rand):
        """
        Performs a selection to choose the next set of children. The number of
        parents will be equal to the number of children. We will also use
        tournament selection with the given size. Further, two children
        will be generated from each parent pairing.

        @param tournament_size {number} The size of the tournament for
            tournament selection.
        @param selection_rand {list} If None, selection will use a random
            number generator. Otherwise, random numbers will be taken in
            sequence from the given list.

        @returns {list of genes} A list of N new genes where N is the number of
            genes in the current population.
        """
        N = len(self.generation)
        new_generation = []
        for n in range(0, int(math.floor(N/2))):
            parents = []
            for p in [0, 1]:
                minindex = -1
                minfitness = 0
                for i in range(0, tournament_size):
                    rdm = self.__get_next_random(selection_rand)
                    index = int(math.floor(rdm*N))
                    fitness = self.generation[index]['fitness']
                    # print '%.4f, %i, %.4f' % (rdm, index, fitness)
                    if minindex == -1 or fitness < minfitness:
                        minfitness = fitness
                        minindex = index
                parents.append(minindex)
            children = self.__crossover(parents, selection_rand)
            # print 'Child 1 = %s' % self.__chromosome2str(children[0])
            # print 'Child 2 = %s' % self.__chromosome2str(children[1])
            new_generation.append(children[0])
            new_generation.append(children[1])
        return new_generation

    def __crossover(self, parents, crossover_rand):
        """
        Performs crossover to create a child of the given parents. If the
        parents are the same, no crossover is performed and the child is
        the parent.

        @param parents {list, size 2} Two indices of parents in self.generation
        @param selection_rand {list} If None, crossover will use a random
            number generator. Otherwise, random numbers will be taken in
            sequence from the given list.

        @returns {list of genes} A gene representing the 2 new children.
        """
        mother = self.generation[parents[0]]['chromosome']
        father = self.generation[parents[1]]['chromosome']
        if parents[0] == parents[1]:
            child1 = deepcopy(mother)
            child2 = deepcopy(father)

        elif self.__get_next_random(crossover_rand) < self.crossover_prob:
            # perform a crossover
            child1 = []
            child2 = []
            for i in range(len(self.variables)):
                crossover = self.__get_next_random(crossover_rand)
                child1.append(crossover*mother[i] +
                              (1-crossover)*father[i])
                child2.append((1 - crossover)*mother[i] +
                              crossover*father[i])
        else:
            child1 = deepcopy(mother)
            child2 = deepcopy(father)

        return child1, child2

    def __mutation(self, new_generation, mutation_rand):
        """
        Performs mutation (with a given probability) on all the new children
        given in new_generation.

        @param new_generation {list of chromosomes} The list of new children
            on which to perform mutation.
        @param mutation_rand {list} If None, mutation will use a random
            number generator. Otherwise, random numbers will be taken in
            sequence from the given list.

        @returns {list of chromosomes} List of child chromosome after mutation
            has been performed.
        """
        for child in new_generation:
            for i in range(len(self.variables)):
                rnd = self.__get_next_random(mutation_rand)
                cond = rnd < self.mutation_prob
                # print '%.4f: %.4f < %.2f => mutate? %r' % (child[i], rnd,
                #                                           self.mutation_prob,
                #                                           cond)
                if cond:
                    j = self.generation_number
                    M = self.total_generations
                    alpha = (1 - (float(j) - 1) / float(M))**float(self.beta)
                    r = self.__get_next_random(mutation_rand)
                    xname = self.variables[i]
                    xmin = self.definition[xname]['lb']
                    xmax = self.definition[xname]['ub']
                    x = child[i]
                    y = xmin + r * (xmax - xmin)
                    if y <= x:
                        z = xmin + ((y - xmin)**alpha) * \
                            ((x - xmin)**(1 - alpha))
                    else:
                        z = xmax - ((xmax - y)**alpha) *\
                            ((xmax - x)**(1 - alpha))

                    # print '\tMutated Gene = %.4f' % z

                    child[i] = z
        return new_generation

    # =========================================================================
    # Private Helpers: Gene and Generation Manipulation
    # =========================================================================

    def __add_chromosome(self, chromosome):
        """
        Adds the given chromosome to the population, computing fi, gi, and g
        for all objectives and constraints i.

        @param chromosome {list} The chromosome.

        @throws {Exception} If the length of chromosome is not the length of
            variables.
        @pre Each entry i in chromosome must correspond to variable i.

        @post {dict} The chromosome representation, given by the following
            key, value pairs:
                'chromosome': The chromosome itself,
                'fi': (where i is an integer > 1) fi(x) where fi is the ith
                    objective and x is the chromosome.
                'gi': (where i is an integer > 1) gi(x) where gi is the ith
                    constraint and g is the chromosome.
                'g': max(0, g1, g2, ...)
        """

        chromosomerep = {'chromosome': chromosome}

        # Compute fi
        i = 1
        for obj in self.objective:
            fi = self.__objective_val(obj, chromosome)
            chromosomerep['f%i' % i] = fi
            i += 1

        # Compute all gi and g = max(0, g1, g2, ...)
        i = 1
        g = 0
        for const in self.ineq:
            gi = self.__constraint_val(const, chromosome)
            chromosomerep['g%i' % i] = gi
            g = max(g, gi)
            i += 1
        for const in self.eq:
            gi = self.__constraint_val(const, chromosome)
            chromosomerep['g%i' % i] = gi
            g = max(g, gi)
            i += 1
        chromosomerep['g'] = g

        self.generation.append(chromosomerep)

    def __compute_fitness(self):
        """
        Computes the fitness Fi for every chromosome in the current generation,
        where for objective i, Fi is defined as
            fi if g = 0
            ffeasmax + g if g > 0
        where ffeasmax is the maximum value of f for all designs where g = 0.

        @pre This should be called at the beginning of every
            __trim_generation().

        @post Every chromosome in self.generation will have Fi added for all
            objectives.
        """
        if len(self.objective) == 1:
            ffeasmax = 0
            for chromosomerep in self.generation:
                f = chromosomerep['f1']
                if is_equal(chromosomerep['g'], 0, 0.0001) and f > ffeasmax:
                    ffeasmax = f

            for chromosomerep in self.generation:
                f = chromosomerep['f1']
                g = chromosomerep['g']
                if is_equal(g, 0, 0.0001):
                    chromosomerep['fitness'] = f
                else:
                    chromosomerep['fitness'] = ffeasmax + g
        else:
            raise Exception('not implemented')

    def __trim_generation(self, should_trim=True):
        """
        Trims the generation, removing the chromosome with the highest fitness
        (which is worse since we are minimizing) until the number of
        chromosomes match the maximum allowed number.

        As a side-effect, also sorts the generation from lowest fitness
        to highest.

        @param should_trim {boolean} If False, only computes fitness, doesn't
            sort or trim. False is useful for sorting.

        @post The size of self.generation is reduced by removing appropriate
            chromosomes.
        """
        self.__compute_fitness()

        if should_trim:
            self.generation = sorted(self.generation,
                                     key=lambda chromosomerep:
                                     chromosomerep['fitness'])
            while len(self.generation) > self.max_gen_size:
                self.generation.pop()

    def __objective_val(self, obj, chromosome):
        """
        Given a chromosome and an objective, returns the objective function
        evaluated at that chromosome.

        @param obj {list} An objective definition.
        @param chromosome {list} A chromosome.

        @returns {number} f(x) where f is defined by the objective and x
            is defined by the chromosome.
        """
        val = obj[0]
        for i in range(1, len(obj)):
            val += obj[i] * chromosome[i-1]
        return val

    def __constraint_val(self, const, chromosome):
        """
        Given a chromosome and a constraint, returns the constraint function
        evaluated at that chromosome.

        @param const {list} An constraint definition.
        @param chromosome {list} A chromosome.

        @returns {number} g(x) where g is defined by the constraint and x
            is defined by the chromosome.
        """
        val = const[0]
        for i in range(1, len(const)):
            val -= const[i] * chromosome[i-1]
        return val

    # =========================================================================
    # Private Helpers: Initialization and Misc.
    # =========================================================================

    def __get_next_random(self, rand_seq):
        """
        If a 'random' sequence is given, pops off the first number of the
        sequence and returns it. Otherwise, returns a randomly generated
        number.

        @param rand_seq {list} The random sequence.

        @returns {number} Either the first entry of rand_seq (popping it off),
            or a randmoly generated number.
        """
        if rand_seq is not None:
            return rand_seq.pop(0)
        else:
            return random.random()

    def __check_variables_definition_feasibility(self):
        """
        Checks the feasibility of the initialized variables and definition.
        """
        for var in self.variables:
            if var not in self.definition:
                raise Exception('%s In variables but not in definition.' % var)
        for var in self.definition:
            if var not in self.variables:
                raise Exception('%s In definition but not in variables.' % var)

    def __check_objective_feasibility(self):
        """
        Checks the feasibility of the objective.
        """
        for obj in self.objective:
            if len(obj) != len(self.variables) + 1:
                raise Exception('Improperly-sized objective')

    def __check_constraints_feasibility(self):
        """
        Checks the feasibility of the constraints.
        """
        for const in self.ineq + self.eq:
            if len(const) != len(self.variables) + 1:
                raise Exception('Improperly-sized constraint(s)')

    def __check_initial_conditions_feasibility(self):
        """
        Checks the feasibility of the initial conditions (starting generation,
        maximum generation size)
        """
        for chromosome in self.starting_gen:
            if len(chromosome) is not len(self.variables):
                raise Exception('Length of chromosome is not equal to ' +
                                'length of defined variables')

    def __initialize_current_generation(self):
        """
        Initializes the current generation according to the starting
        generation.
        """
        self.generation = []
        for chromosome in self.starting_gen:
            self.__add_chromosome(chromosome)
        self.__trim_generation(self.trim_first)
        self.generation_number = 1

    def __populate_definition(self, definition):
        """
        Populates definition with defaults for any parameter that
        were not given for each variable.

        @param definition {dict} The problem definition where keys are variable
            names and values are the definition for the variables. See the
            @post condition for allowable definition values.

        @returns {dict} A populated definition where the key is
            the variable name and the value is another dictionary with
            the following keys (additional keys are ignored):
                lb {number} (default = -infinity) The lower bound on allowable
                    values the variable can take
                ub {number} (default = infinity) The upper bound on allowable
                    values the variable can take
                category {string in {'discreet', 'continuous'}}
                    (default = 'continuous') The type of the variable, whether
                    discreet or continuous.
        """
        new_definition = {}
        for (var, defn) in definition.iteritems():
            new_definition[var] = {
                'lb': defn.get('lb', float('-inf')),
                'ub': defn.get('ub', float('inf')),
                'category': defn.get('category', 'continuous')
            }
        return new_definition

    def __definition2str(self, defn):
        """
        Takes a definition for a particular variable and converts it into
        a string.

        @param defn {dict} The problem definition.

        @returns {string} A string representation of the definition.
        """
        return '(lb=%s, ub=%s, category=%s)' % (defn['lb'], defn['ub'],
                                                defn['category'])

    def __objective2str(self, obj):
        """
        Takes an objective and converts it to a string.

        @param obj {list} The definition for an objective.

        @returns {string} A string representation of the objective.
        """
        strng = 'minimize %.4s' % obj[0]
        for i in range(1, len(obj)):
            if not is_equal(obj[i], 0, 0.0001):
                strng += ' + (%.4s)%s' % (obj[i], self.variables[i - 1])
        return strng

    def __const2str(self, const, comp):
        """
        Takes a constraint and converts it into a string.

        @param const {list} The definition for a constraint.
        @param conp {string in {'<=', '=='}} The symbol definining the type
            of constraint.

        @returns {string} A string representation of the constraint.
        """
        strng = '%.4s' % const[0]
        for i in range(1, len(const)):
            if not is_equal(const[i], 0, 0.0001):
                strng += ' - (%.4s)%s' % (const[i], self.variables[i-1])
        strng += ' %s 0' % comp
        return strng

    def __chromosome2str(self, chromosome):
        """
        Takes a chromosome and converts it into a string.

        @param chromosome {iterable} The array/list/etc. defining the
            chromosme.

        @returns {string} A string representation of the chromosome.
        """
        strng = '['
        first = True
        index = 0
        for var in chromosome:
            if not first:
                strng = '%s, ' % strng
            strng = '%s%s=%s' % (strng, self.variables[index], var)
            first = False
            index += 1
        strng = '%s]' % strng
        return strng
