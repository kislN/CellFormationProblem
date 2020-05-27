import numpy as np
import copy

'''
mine  article's  meaning 
S     S          current solution
S_c   S^c        neighborhood solution
S_1   S*         best solution found in current number of cells
S_2   S**        best solution found so far
T_0   T_0        initial temperature
T_f   T_f        final temperature
alpha alpha          cooling rate
L     L          Markov chain length
k     k          iteration number
C     C          initial number of cells
C_1   C*         optimal number of cells
D     D          length of period for evoking exchange-move
'''


class Annealing:

    def __init__(self, adj_matrix, machines=None, parts=None):
        self.adj_matrix = adj_matrix
        self.machines = machines
        self.parts = parts
        self.m = adj_matrix.shape[0]
        self.p = adj_matrix.shape[1]
        self.n_ones = adj_matrix.sum()
        self.n_zeros = self.m * self.p - self.n_ones
        self.D = None
        self.similar_list = self.get_similar_list()

    def get_similar_list(self):
        similar_matrix = np.ones((self.p, self.p))
        similar_dict = {}
        for i in range(self.p):
            for j in range(i + 1, self.p):
                set_i = set(np.where(self.adj_matrix[:, i] == 1)[0])
                set_j = set(np.where(self.adj_matrix[:, j] == 1)[0])
                similar_matrix[i, j] = similar_matrix[j, i] = similar_dict[(i, j)] = \
                    len(set_i & set_j) / (len(set_i & set_j) + len(set_i - set_j) + len(set_j - set_i))
        similar_list = sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)
        return similar_list

    def obj_function(self, n1_out, n0_in):
        return (self.n_ones - n1_out) / (self.n_ones + n0_in)

    # def obj_function(self, S):
    #     p_clusters = S[0]
    #     m_clusters = S[1]
    #     n_clust = S[2]
    #     p_clust_matrix = []         # TODO: check if just for works faster then sets
    #     for i in range(n_clust):
    #         p_clust_matrix.append(set(np.where(p_clusters == i)[0]))
    #     n1_out = 0
    #     n0_in = 0
    #     for machine in range(self.m):
    #         machine_parts = self.machines[machine]
    #         cluster_parts = p_clust_matrix[m_clusters[machine]]
    #         n1_out += len(machine_parts - cluster_parts)
    #         n0_in += len(cluster_parts - machine_parts)
    #     return (self.n_ones - n1_out)/(self.n_ones + n0_in)

    def get_m_clusters(self, n_clust, p_clusters):
        p_clust_matrix = []
        for i in range(n_clust):
            p_clust_matrix.append(set(np.where(p_clusters == i)[0]))
        m_clusters = np.asarray([None] * self.m)
        n1_out = 0
        n0_in = 0
        for machine in range(self.m):
            min_ve = self.p + 1
            min_ve_clust = 0
            for clust in range(n_clust):
                ve = len(p_clust_matrix[clust] - self.machines[machine]) + \
                     len(self.machines[machine] - p_clust_matrix[clust])
                if min_ve > ve:
                    min_ve = ve
                    min_ve_clust = clust
            m_clusters[machine] = min_ve_clust
            n1_out += len(self.machines[machine] - p_clust_matrix[min_ve_clust])
            n0_in += len(p_clust_matrix[min_ve_clust] - self.machines[machine])
        return [p_clusters, m_clusters, n1_out, n0_in]

    def initial_solution(self, n_clust):
        p_clusters = np.array([None] * self.p)
        num_cluster = 0
        p_counter = 0
        for p_pair in self.similar_list:
            if p_counter != self.p:
                if p_clusters[p_pair[0][0]] is None and p_clusters[p_pair[0][1]] is None:
                    p_clusters[p_pair[0][0]] = p_clusters[p_pair[0][1]] = num_cluster
                    num_cluster = (num_cluster + 1) % n_clust
                    p_counter += 2
                elif p_clusters[p_pair[0][1]] is None:
                    p_clusters[p_pair[0][1]] = p_clusters[p_pair[0][0]]
                    p_counter += 1
                elif p_clusters[p_pair[0][0]] is None:
                    p_clusters[p_pair[0][0]] = p_clusters[p_pair[0][1]]
                    p_counter += 1
            else:
                break
                                                    # TODO: add filling the empty clusters
        # n_clust = np.max(p_clusters) + 1
        return self.get_m_clusters(n_clust, p_clusters)

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


    def single_move(self, n_clust, S):
        curr_obj = self.obj_function(S[2], S[3])
        best_solution = None
        destination_cluster = None
        max_delta = -curr_obj
        for part in range(self.p):
            clusters = list(range(n_clust))
            clusters.pop(S[0][part])
            for cluster in clusters:
                new_S = self.single_move_step(part, cluster, S)
                delta_obj = self.obj_function(new_S[2], new_S[3]) - curr_obj
                if delta_obj > max_delta:
                    max_delta = delta_obj
                    best_solution = new_S
                    destination_cluster = cluster
        return best_solution, destination_cluster       # TODO: leave it if exchange_move works with this function

    def exchange_move(self, n_clust, S):       # TODO: test this fuction
        curr_obj = self.obj_function(S[2], S[3])
        best_solution = None
        max_delta = -curr_obj
        for part in range(self.p):
            clusters = list(range(n_clust))
            clusters.pop(S[0][part])
            for cluster in clusters:
                exchange_parts = np.where(S[0] == cluster)[0]
                new_S = self.single_move_step(part, cluster, S)
                for exchange_p in exchange_parts:
                    second_new_S = self.single_move_step(exchange_p, cluster, new_S)
                    delta_obj = self.obj_function(second_new_S[2], second_new_S[3]) - curr_obj
                    if delta_obj > max_delta:
                        max_delta = delta_obj
                        best_solution = second_new_S
        return best_solution

    def generate_neighbor(self, S, C, counter):  # TODO: self.C, self.counter (if it will be possible)
        new_S, d_c = self.single_move(C, S)
        # if counter % self.D == 0:
        if 1:
            new_S = self.exchange_move(C, new_S)
        new_S = self.get_m_clusters(C, new_S)
        return new_S


    def run(self, init_n_clust=2, T_0=10, T_f=0.002, alpha=0.7, L=10, D=18, ):
        self.D = D
        counter = counter_MC = counter_trapped = counter_stagnant = 0

        C_1 = C = 2
        S = self.initial_solution(C)
        S_2 = copy.deepcopy(S)
        S_1 = copy.deepcopy(S)

        if counter_MC < L and counter_trapped < L/2:
            S_c = self.generate_neighbor(S, C, counter)

        print()




    def update_temperature(self, a):
        self.temperature *= a







