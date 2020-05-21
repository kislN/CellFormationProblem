import numpy as np

class Annealing:

    def __init__(self, matrix, temperature=0.8):
        self.matrix = matrix    # numpy array
        self.temperature = temperature
        self.m = matrix.shape[0]
        self.p = matrix.shape[1]

    def initial_solution(self):
        B = np.ones((self.p, self.p))
        for i in range(self.p):
            for j in range(i+1, self.p):
                set_i = set(np.where(self.matrix[:, i] == 1)[0])
                set_j = set(np.where(self.matrix[:, j] == 1)[0])
                B[i, j] = B[j, i] = len(set_i & set_j) / \
                                    (len(set_i & set_j) + len(set_i - set_j) + len(set_j - set_i))

    def update_temperature(self, a):
        self.temperature *= a







