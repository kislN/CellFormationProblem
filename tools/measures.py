import numpy as np
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
    df = pd.DataFrame(columns=['case', 'C', 'T0', 'Tf', 'alpha', 'L', 'D', 'check', 'mean_time', 'efficacy', 'clusters'])
    C = 2
    check = 4
    T_f = 0.002
    iters = 1
    for case in tqdm(cases):
        for T_0 in [10]: #[10, 30, 50]:
            for alpha in [0.7]: #[0.7, 0.8, 0.9]:
                for L in [10]: #[10, 30, 70]:
                    for D in [6]: #[6, 12, 18]:
                        result = get_result(case, iterations=iters, C=C, T_0=T_0, T_f=T_f, alpha=alpha, L=L, D=D, check=check)
                        df = df.append(pd.Series([case['file_name'], C, T_0, T_f, alpha, L, D, check, result[0],
                                                  result[1], result[2]], index=df.columns), ignore_index=True)
    return df