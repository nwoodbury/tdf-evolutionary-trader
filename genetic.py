"""
An implementation of a multi-objective value-based genetic algorithm.
"""

from termcolor import colored
import random
import math


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

    def __init__(self, name, variables, definition, objective, ineq, eq,
                 starting_gen, max_gen_size=None, trim_first=True):
        """
        Algorithm initialization.

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
        # name initialization
        self.name = name

        # variables and problem definition initialization
        self.variables = variables
        self.definition = self.__populate_definition(definition)
        for var in self.variables:
            if var not in self.definition:
                raise Exception('%s In variables but not in definition.' % var)
        for var in self.definition:
            if var not in self.variables:
                raise Exception('%s In definition but not in variables.' % var)

        # objective initialization
        self.objective = objective
        for obj in objective:
            if len(obj) != len(self.variables) + 1:
                raise Exception('Improperly-sized objective')

        # constraints (ineq, eq) initialization
        self.ineq = ineq
        self.eq = eq
        for const in self.ineq + self.eq:
            if len(const) != len(self.variables) + 1:
                raise Exception('Improperly-sized constraint(s)')

        # initial conditions (starting_gen) initialization
        self.starting_gen = starting_gen
        for gene in self.starting_gen:
            if len(gene) is not len(variables):
                raise Exception('Length of gene is not equal to length of ' +
                                'defined variables')

        # maximum generation size
        if max_gen_size is None:
            max_gen_size = len(starting_gen)
        self.max_gen_size = max_gen_size

        # current generation
        self.generation = []
        for gene in self.starting_gen:
            self.__add_gene(gene)
        self.__trim_generation(trim_first)

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
        for gene in self.starting_gen:
            strng += '\t%s\n' % self.__gene2str(gene)

        strng += '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        return strng

    def current_generation_str(self, verbose=False):
        """
        Returns a string representation of the current generation.

        @param verbose {boolean} If true, shows all fi and gi.

        @returns {string}
        """
        print '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
        print 'Current generation for Problem: %s\n' % self.name

        j = 1
        for generep in self.generation:
            infeasible_string = ''
            if not is_equal(generep['g'], 0, 0.0001):
                infeasible_string = colored(' (Infeasible)', 'red')
            print 'Design %i%s: %s' % (j, infeasible_string,
                                       self.__gene2str(generep['gene']))

            print colored('\tfitness = %.4f' % generep['fitness'], 'green')

            if verbose:
                for i in range(1, len(self.objective) + 1):
                    print '\tf%i = %.4f' % (i, generep['f%i' % i])
                for i in range(1, len(self.ineq) + len(self.eq) + 1):
                    print '\tg%i = %.4f' % (i, generep['g%i' % i])
                print '\tg = %.4f' % generep['g']

            j += 1

        print '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'

    def next_generation(self, tournament_size=2,
                        selection_rand=None, mutation_rand=None):
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

        @returns {boolean} True if the next generation is different from the
            current generation.
        """
        new_generation = self.__selection(tournament_size, selection_rand)
        print new_generation

    #==========================================================================
    # Private Helpers: Selection, Crossover, and Mutation
    #==========================================================================

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

        @returns {list} A list of N new genes where N is the number of genes
            in the current population.
        """
        N = len(self.generation)
        for n in range(0, int(math.floor(N/2))):
            parents = []
            for p in [0, 1]:
                minindex = -1
                minfitness = 0
                for i in range(0, tournament_size):
                    if selection_rand is None:
                        rdm = random.random()
                    else:
                        rdm = selection_rand.pop(0)
                    index = int(math.floor(rdm*N))
                    fitness = self.generation[index]['fitness']
                    print '%.4f, %i, %.4f' % (rdm, index, fitness)
                    if minindex == -1 or fitness < minfitness:
                        minfitness = fitness
                        minindex = index
                parents.append(minindex)
            print parents

    #==========================================================================
    # Private Helpers: Gene and Generation Manipulation
    #==========================================================================

    def __add_gene(self, gene):
        """
        Adds the given gene to the population, computing fi, gi, and g for all
        objectives and constraints i.

        @param gene {list} The gene.

        @throws {Exception} If the length of gene is not the length of
            variables.
        @pre Each entry i in gene must correspond to variable i.

        @post {dict} The gene representation, given by the following key, value
            pairs:
                'gene': The gene itself,
                'fi': (where i is an integer > 1) fi(x) where fi is the ith
                    objective and x is the gene.
                'gi': (where i is an integer > 1) gi(x) where gi is the ith
                    constraint and g is the gene.
                'g': max(0, g1, g2, ...)
        """

        generep = {'gene': gene}

        # Compute fi
        i = 1
        for obj in self.objective:
            fi = self.__objective_val(obj, gene)
            generep['f%i' % i] = fi
            i += 1

        # Compute all gi and g = max(0, g1, g2, ...)
        i = 1
        g = 0
        for const in self.ineq:
            gi = self.__constraint_val(const, gene)
            generep['g%i' % i] = gi
            g = max(g, gi)
            i += 1
        for const in self.eq:
            gi = self.__constraint_val(const, gene)
            generep['g%i' % i] = gi
            g = max(g, gi)
            i += 1
        generep['g'] = g

        self.generation.append(generep)

    def __compute_fitness(self):
        """
        Computes the fitness Fi for every gene in the current generation,
        where for objective i, Fi is defined as
            fi if g = 0
            ffeasmax + g if g > 0
        where ffeasmax is the maximum value of f for all designs where g = 0.

        @pre This should be called at the beginning of every
            __trim_generation().

        @post Every gene in self.generation will have Fi added for all
            objectives.
        """
        if len(self.objective) == 1:
            ffeasmax = 0
            for generep in self.generation:
                f = generep['f1']
                if is_equal(generep['g'], 0, 0.0001) and f > ffeasmax:
                    ffeasmax = f

            for generep in self.generation:
                f = generep['f1']
                g = generep['g']
                if is_equal(g, 0, 0.0001):
                    generep['fitness'] = f
                else:
                    generep['fitness'] = ffeasmax + g
        else:
            raise Exception('not implemented')

    def __trim_generation(self, should_trim=True):
        """
        Trims the generation, removing the gene with the highest fitness
        (which is worse since we are minimizing) until the number of genes
        match the maximum allowed number.

        As a side-effect, also sorts the generation from lowest fitness
        to highest.

        @param should_trim {boolean} If False, only computes fitness, doesn't
            sort or trim. False is useful for sorting.

        @post The size of self.generation is reduced by removing appropriate
            genes.
        """
        self.__compute_fitness()

        if should_trim:
            self.generation = sorted(self.generation,
                                     key=lambda generep: generep['fitness'])
            while len(self.generation) > self.max_gen_size:
                self.generation.pop()

    def __objective_val(self, obj, gene):
        """
        Given a gene and an objective, returns the objective function
        evaluated at that gene.

        @param obj {list} An objective definition.
        @param gene {list} A gene.

        @returns {number} f(x) where f is defined by the objective and x
            is defined by the gene.
        """
        val = obj[0]
        for i in range(1, len(obj)):
            val += obj[i] * gene[i-1]
        return val

    def __constraint_val(self, const, gene):
        """
        Given a gene and a constraint, returns the constraint function
        evaluated at that gene.

        @param const {list} An constraint definition.
        @param gene {list} A gene.

        @returns {number} g(x) where g is defined by the constraint and x
            is defined by the gene.
        """
        val = const[0]
        for i in range(1, len(const)):
            val -= const[i] * gene[i-1]
        return val

    #==========================================================================
    # Private Helpers: Initialization
    #==========================================================================

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

        @returns {string} A string representation of the gene.
        """
        strng = '%.4s' % const[0]
        for i in range(1, len(const)):
            if not is_equal(const[i], 0, 0.0001):
                strng += ' - (%.4s)%s' % (const[i], self.variables[i-1])
        strng += ' %s 0' % comp
        return strng

    def __gene2str(self, gene):
        """
        Takes a gene and converts it into a string.

        @param gene {iterable} The array/list/etc. defining the gene.

        @returns {string} A string representation of the gene.
        """
        strng = '['
        first = True
        index = 0
        for var in gene:
            if not first:
                strng = '%s, ' % strng
            strng = '%s%s=%s' % (strng, self.variables[index], var)
            first = False
            index += 1
        strng = '%s]' % strng
        return strng
