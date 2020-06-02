from tools.data_loader import get_data
import os
from algorithms.simulated_annealing import Annealing as SA
import numpy as np
from tools.measures import get_all_results
from tools.check_result import check, get_res


np.random.seed(41)

files = os.listdir('./data/cfp_data')
files.sort()

cases = []
for file in files:
    cases.append(get_data(file))

for n in range(5):
    m, p = get_res('./data/solutions/' + cases[n]['file_name'][:5] + '.sol')
    mat, eff = check(cases[n]['incidence_matrix'], m, p)
    print(cases[n]['file_name'][:5], ': ', eff)




