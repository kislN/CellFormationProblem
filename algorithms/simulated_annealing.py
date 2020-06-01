import numpy as np
import copy
from tools.decorators import timeit

'''
mine     article's  meaning 
S        S          current solution
S_c      S^c        neighborhood solution
S_1      S*         best solution found in current number of cells
S_best   S**        best solution found so far
T_0      T_0        initial temperature
T_f      T_f        final temperature
alpha    alpha      cooling rate
L        L          Markov chain length
k        k          iteration number
C        C          initial number of cells
C_best   C*         optimal number of cells
D        D          length of period for evoking exchange-move
'''


class Annealing:

    def __init__(self, incidence_matrix, machines, parts):
        self.incidence_matrix = incidence_matrix
        self.machines = machines
        self.parts = parts
        self.m = incidence_matrix.shape[0]
        self.p = incidence_matrix.shape[1]
        self.n_ones = incidence_matrix.sum()
        self.similar_list = self.get_similar_list()

        self._C = None
        self.C_best = None

        self._S = None
        self._S_1 = None
        self.S_best = None
        self.obj_best = None

        self.T_k = None
        self.L = None
        self.D = None

        self._counter = None
        self._counter_MC = None
        self._counter_trapped = None
        self._counter_stagnant = None

    def obj_function(self, n1_out, n0_in):
        return (self.n_ones - n1_out) / (self.n_ones + n0_in)

    def get_similar_list(self):
        similar_dict = {}
        for i in range(self.p):
            set_i = self.parts[i]
            for j in range(i + 1, self.p):
                set_j = self.parts[j]
                similar_dict[(i, j)] = len(set_i & set_j) / \
                                       (len(set_i & set_j) + len(set_i - set_j) + len(set_j - set_i))
        similar_list = sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)
        return similar_list

    def initial_solution(self):
        p_clusters = np.array([None] * self.p)
        num_cluster = 0
        p_counter = 0
        for p_pair in self.similar_list:
            if p_counter != self.p:
                if p_clusters[p_pair[0][0]] is None and p_clusters[p_pair[0][1]] is None:
                    p_clusters[p_pair[0][0]] = p_clusters[p_pair[0][1]] = num_cluster
                    num_cluster = (num_cluster + 1) % self._C
                    p_counter += 2
                elif p_clusters[p_pair[0][1]] is None:
                    p_clusters[p_pair[0][1]] = p_clusters[p_pair[0][0]]
                    p_counter += 1
                elif p_clusters[p_pair[0][0]] is None:
                    p_clusters[p_pair[0][0]] = p_clusters[p_pair[0][1]]
                    p_counter += 1
            else:
                break

        n_clust = np.max(p_clusters) + 1
        if n_clust != self._C:
            used_parts = []
            i = 0
            for cluster in range(n_clust, self._C):
                while True:
                    part = self.similar_list[-(i + 1)][0][0]
                    if part in used_parts:
                        i += 1
                        continue
                    p_clusters[part] = cluster
                    used_parts.append(part)
                    break

        if np.size(np.unique(p_clusters)) != self._C:
            print('oooooh sheeet')

        return self.get_m_clusters(p_clusters)

    def get_m_clusters(self, p_clusters):
        p_clust_matrix = []
        for i in range(self._C):
            p_clust_matrix.append(set(np.where(p_clusters == i)[0]))

        m_clusters = np.asarray([None] * self.m)
        matrix_ve = np.zeros((self.m, self._C))
        clust_count = np.zeros((self._C))
        for machine in range(self.m):
            min_ve = self.p + 1
            min_ve_clust = None
            for clust in range(self._C):
                ve = len(p_clust_matrix[clust] - self.machines[machine]) + \
                                            len(self.machines[machine] - p_clust_matrix[clust])
                matrix_ve[machine, clust] = ve
                if min_ve > ve:
                    min_ve = ve
                    min_ve_clust = clust
            m_clusters[machine] = min_ve_clust
            clust_count[min_ve_clust] += 1

        diff_clust = np.setdiff1d(np.arange(self._C), np.unique(m_clusters))
        if np.size(diff_clust):
            for clust in diff_clust:
                while True:
                    min_ve_machine = np.argmin(matrix_ve.T[clust])
                    matrix_ve[min_ve_machine, clust] = self.p + 1
                    if clust_count[m_clusters[min_ve_machine]] > 1:
                        clust_count[m_clusters[min_ve_machine]] -= 1
                        m_clusters[min_ve_machine] = clust
                        clust_count[clust] += 1
                        break

        n1_out = 0
        n0_in = 0
        for machine in range(self.m):
            n1_out += len(self.machines[machine] - p_clust_matrix[m_clusters[machine]])
            n0_in += len(p_clust_matrix[m_clusters[machine]] - self.machines[machine])

        if np.size(np.unique(m_clusters)) != self._C:
            print('ooooh fuck')
        return [p_clusters, m_clusters, n1_out, n0_in]

    def single_move_step(self, part, new_cluster, S):
        p_clusters = copy.deepcopy(S[0])
        m_clusters = copy.deepcopy(S[1])
        n1_out = S[2]
        n0_in = S[3]

        current_clust = p_clusters[part]
        current_clust_machines = set(np.where(m_clusters == current_clust)[0])
        new_clust_machines = set(np.where(m_clusters == new_cluster)[0])
        p_clusters[part] = new_cluster

        n1_out = n1_out + len(self.parts[part] & current_clust_machines) - len(self.parts[part] & new_clust_machines)
        n0_in = n0_in + len(new_clust_machines - self.parts[part]) - len(current_clust_machines - self.parts[part])
        return [p_clusters, m_clusters, n1_out, n0_in]

    def single_move(self, S):
        curr_obj = self.obj_function(S[2], S[3])
        best_solution = None
        source = None
        destin = None
        max_delta = -curr_obj
        for part in range(self.p):
            clusters = list(range(self._C))
            s_clust = clusters.pop(S[0][part])
            for cluster in clusters:
                new_S = self.single_move_step(part, cluster, S)
                delta_obj = self.obj_function(new_S[2], new_S[3]) - curr_obj
                if delta_obj > max_delta:
                    max_delta = delta_obj
                    best_solution = new_S
                    source = s_clust
                    destin = cluster
                    b_part = part
        return best_solution, b_part, source, destin

    # def exchange_move(self, S):           # the longest version
    #     curr_obj = self.obj_function(S[2], S[3])
    #     best_solution = None
    #     max_delta = -curr_obj
    #     for part in range(self.p):
    #         clusters = list(range(self._C))
    #         source_clust = clusters.pop(S[0][part])
    #         for cluster in clusters:
    #             exchange_parts = np.where(S[0] == cluster)[0]
    #             new_S = self.single_move_step(part, cluster, S)
    #             for exchange_p in exchange_parts:
    #                 second_new_S = self.single_move_step(exchange_p, source_clust, new_S)
    #                 delta_obj = self.obj_function(second_new_S[2], second_new_S[3]) - curr_obj
    #                 if delta_obj > max_delta:
    #                     max_delta = delta_obj
    #                     best_solution = second_new_S
    #     return best_solution

    # def exchange_move(self, S):
    #     new_S, b_part, source, destin = self.single_move(S)
    #     curr_obj = self.obj_function(new_S[2], new_S[3])
    #     best_solution = None
    #     max_delta = -curr_obj
    #     exchange_parts = np.where(new_S[0] == destin)[0]
    #     for part in exchange_parts:
    #         if part != b_part:
    #             new_new_S = self.single_move_step(part, source, new_S)
    #             delta_obj = self.obj_function(new_new_S[2], new_new_S[3]) - curr_obj
    #             if delta_obj > max_delta:
    #                 max_delta = delta_obj
    #                 best_solution = new_new_S
    #     return best_solution

    def exchange_move(self, S, swapped_part, source, destin):
        curr_obj = self.obj_function(S[2], S[3])
        best_solution = None
        max_delta = -curr_obj
        exchange_parts = np.where(S[0] == destin)[0]
        for part in exchange_parts:
            if part != swapped_part:
                new_S = self.single_move_step(part, source, S)
                delta_obj = self.obj_function(new_S[2], new_S[3]) - curr_obj
                if delta_obj > max_delta:
                    max_delta = delta_obj
                    best_solution = new_S
        return best_solution

    def generate_neighbor(self, S):
        new_S, part, source, destin = self.single_move(S)
        if self._counter % self.D == 0 or np.size(np.unique(new_S[0])) != self._C:
            new_S = self.exchange_move(new_S, part, source, destin)

        if np.size(np.unique(new_S[0])) != self._C:
            print('ohhhhhh noooo')

        new_S = self.get_m_clusters(new_S[0])

        if np.size(np.unique(new_S[0])) != np.size(np.unique(new_S[1])):
            print('You are asshole')

        return new_S

    def inside_loop(self):
        while self._counter_MC < self.L and self._counter_trapped < self.L/2:
            S_c = self.generate_neighbor(self._S)

            obj_S_c = self.obj_function(S_c[2], S_c[3])
            obj_S_1 = self.obj_function(self._S_1[2], self._S_1[3])

            if obj_S_c > obj_S_1:
                self._S_1 = copy.deepcopy(S_c)
                self._S = copy.deepcopy(S_c)
                self._counter_stagnant = 0
                self._counter_MC += 1
                break

            elif obj_S_c == obj_S_1:
                self._S = copy.deepcopy(S_c)
                self._counter_stagnant += 1
                self._counter_MC += 1
                break

            delta = obj_S_c - self.obj_function(self._S[2], self._S[3])
            x = np.random.randint(2)
            if np.exp(delta/self.T_k) > x:
                self._S = copy.deepcopy(S_c)
                self._counter_trapped = 0
            else:
                self._counter_trapped += 1

            self._counter_MC += 1

    # @timeit
    def run(self, C=2, T_0=10, T_f=0.002, alpha=0.7, L=10, D=18, check=4):
        self._C = C

        self.L = L
        self.D = D

        self._S = self.initial_solution()
        self._S_1 = copy.deepcopy(self._S)

        self.S_best = copy.deepcopy(self._S_1)
        self.obj_best = self.obj_function(self.S_best[2], self.S_best[3])
        self.C_best = self._C

        while True:             # for step 5
            self._counter = 0
            self._counter_trapped = 0
            self._counter_stagnant = 0
            self.T_k = T_0

            while True:         # for step 4
                self._counter_MC = 0
                self.inside_loop()
                if self.T_k > T_f and self._counter_stagnant < check:
                    self.T_k *= alpha
                    self._counter += 1
                else:
                    break

            if self.obj_function(self._S_1[2], self._S_1[3]) > self.obj_best:
                self.S_best = copy.deepcopy(self._S_1)
                self.obj_best = self.obj_function(self.S_best[2], self.S_best[3])
                self.C_best = self._C
                self._C += 1
                if self._C > self.m:
                    break
                self._S = self.initial_solution()        # with new number of clusters
                self._S_1 = copy.deepcopy(self._S)
            else:
                break