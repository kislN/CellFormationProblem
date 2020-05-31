import numpy as np
from os.path import join
import pandas as pd
import time
from tqdm import tqdm

from algorithms.simulated_annealing import Annealing as SA


def get_result(case, iterations=3, **params):
    case_class = SA(case['incidence_matrix'], machines=case['machines'], parts=case['parts'])
    delta_time = 0
    efficacy = 0
    for _ in range(iterations):
        ts = time.time()
        case_class.run(**params)
        te = time.time()
        delta_time += te - ts
        if case_class.obj_best > efficacy:
            efficacy = case_class.obj_best
            clusters = case_class.C_best
            solution = case_class.S_best
    delta_time = round(delta_time / iterations, 8)
    return (delta_time, efficacy, clusters, solution)

def get_all_results(cases):
    path = './data'
    df = pd.DataFrame(columns=['case', 'C', 'T0', 'Tf', 'alpha', 'L', 'D', 'check', 'mean_time', 'efficacy', 'clusters'])
    C = 2
    check = 4
    T_f = 0.002
    iters = 1
    for case in tqdm(cases):
        best_effic = 0
        best_clusters = 0
        best_solution = None
        for T_0 in [10]: #[10, 30, 50]:
            for alpha in [0.7]: #[0.7, 0.8, 0.9]:
                for L in [10]: #[10, 30, 70]:
                    for D in [18]: #[6, 12, 18]:
                        result = get_result(case, iterations=iters, C=C, T_0=T_0, T_f=T_f, alpha=alpha, L=L, D=D, check=check)
                        df = df.append(pd.Series([case['file_name'], C, T_0, T_f, alpha, L, D, check, result[0],
                                                  result[1], result[2]], index=df.columns), ignore_index=True)
                        if best_effic < result[1]:
                            best_effic = result[1]
                            best_clusters = result[2]
                            best_solution = result[3]

        file_name = case['file_name'][:5] + '.sol'
        sol_str = ''
        for i in range(case['m']):
            sol_str += str(i + 1) + '_' + str(best_solution[1][i]) + ' '
        sol_str += '\n'
        for i in range(case['p']):
            sol_str += str(i + 1) + '_' + str(best_solution[0][i]) + ' '
        with open(join(path, 'solutions', file_name), 'w') as file:
            file.write(sol_str)

    df.to_csv(join(path, 'results.csv'))

    return df