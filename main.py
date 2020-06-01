from tools.data_loader import get_data
import os
from algorithms.simulated_annealing import Annealing as SA
import numpy as np
from tools.measures import get_all_results
from tools.check_result import check

np.random.seed(41)

files = os.listdir('./data/cfp_data')
files.sort()

cases = []
for file in files:
    cases.append(get_data(file))

df = get_all_results(cases)

# n = 3
# test = SA(cases[n]['incidence_matrix'], machines=cases[n]['machines'], parts=cases[n]['parts'])
# test.run(C=11)
#
# mat, eff = check(cases[n]['incidence_matrix'], test.S_best[1], test.S_best[0])

print()
print(df)


