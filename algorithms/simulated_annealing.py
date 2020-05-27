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
        self.adj_matrix = adj_matrix    # numpy array
        self.machines = machines
        self.parts = parts
        self.m = adj_matrix.shape[0]
        self.p = adj_matrix.shape[1]
        self.n_ones = adj_matrix.sum()
        self.n_zeros = self.m * self.p - self.n_ones

    def get_m_clusters(self, n_clust, p_clusters):
        p_clust_matrix = []
        for i in range(n_clust):
            p_clust_matrix.append(set(np.where(p_clusters == i)[0]))

        m_clusters = np.asarray([None] * self.m)
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
        return m_clusters

    def initial_solution(self, n_clust):
        similar_matrix = np.ones((self.p, self.p))
        similar_dict = {}
        for i in range(self.p):
            for j in range(i+1, self.p):
                set_i = set(np.where(self.adj_matrix[:, i] == 1)[0])
                set_j = set(np.where(self.adj_matrix[:, j] == 1)[0])
                similar_matrix[i, j] = similar_matrix[j, i] = similar_dict[(i ,j)] = \
                    len(set_i & set_j) / (len(set_i & set_j) + len(set_i - set_j) + len(set_j - set_i))
        similar_list = sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)

        p_clusters = np.array([None] * self.p)
        num_cluster = 0
        p_counter = 0
        for p_pair in similar_list:
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

        n_clust = np.max(p_clusters) + 1
        m_clusters = self.get_m_clusters(n_clust, p_clusters)

        return [p_clusters, m_clusters]

    def generate_neighbor(self, S):
        pass

    def obj_function(self, S):
        p_clust_matrix = []         # TODO: check if just for works faster then sets
        for i in range(n_clust):
            p_clust_matrix.append(set(np.where(p_clusters == i)[0]))


    def run(self, init_n_clust=2, T_0=10, T_f=0.002, alpha=0.7, L=10, D=18, ):
        S = self.initial_solution(init_n_clust)
        S_2 = copy.deepcopy(S)
        S_1 = copy.deepcopy(S)
        C_1 = C = 2
        counter = counter_MC = counter_trapped = counter_stagnant = 0

        if counter < L and counter_trapped < L/2:
            new = self.generate_neighbor(S)

        print()




    def update_temperature(self, a):
        self.temperature *= a







