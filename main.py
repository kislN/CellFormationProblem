from tools.data_loader import get_data
import os
from algorithms.simulated_annealing import Annealing as SA

files = os.listdir('./data/cfp_data')
files.sort()

cases = []
for file in files:
    cases.append(get_data(file))
print()

test = SA(cases[5]['adj_matrix'], machines=cases[5]['machines'], parts=cases[5]['parts'])
# test.initial_solution(n_clust=2)
test.run(init_n_clust=2)

print()

