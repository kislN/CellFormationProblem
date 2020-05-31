from tools.data_loader import get_data
import os
from algorithms.simulated_annealing import Annealing as SA
import numpy as np
from tools.measures import get_all_results
from tools.check_result import form_matrix

np.random.seed(41)

files = os.listdir('./data/cfp_data')
files.sort()

cases = []
for file in files:
    cases.append(get_data(file))

# df = get_all_results(cases)

n = 4
test = SA(cases[n]['incidence_matrix'], machines=cases[n]['machines'], parts=cases[n]['parts'])
test.run()

s1 = '1 1 2 2 1 1 2 2 2 2 2 2 2 2 1 2 2 1 1 2 1 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1'
s2 = '2 1 1 1 2 2 1 2 1 1 1 1 1 2 1 2 1 2 1 1 1 2 1 2 2 2 2 2 2 1 1 2 2 1 1 2 2 2 2 2 1 2 2 1 1 2 1 2 2 2 2 2 2'

m = np.asarray(s1.split(' '), dtype=int) - 1
p = np.asarray(s2.split(' '), dtype=int) - 1

mat, eff = form_matrix(cases[n]['incidence_matrix'], test.S_best[1], test.S_best[0])

# mat2, eff2 = form_matrix(cases[n]['incidence_matrix'], m, p)

print()
# print(df)


