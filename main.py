from tools.data_loader import get_data
import os
from algorithms.simulated_annealing import Annealing as SA
import numpy as np
from tools.measures import get_all_results

# np.random.seed(41)

files = os.listdir('./data/cfp_data')
files.sort()

cases = []
for file in files:
    cases.append(get_data(file))

# n = 1
# test = SA(cases[n]['incidence_matrix'], machines=cases[n]['machines'], parts=cases[n]['parts'])
#
# test.run()
# print(test.C_best)
# print(test.obj_best)

df = get_all_results(cases)

print(df)

print()

