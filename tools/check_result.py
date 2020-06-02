import numpy as np

def check(ind_matrix, m_clusters, p_clusters):
    if np.size(np.unique(m_clusters)) != np.size(np.unique(p_clusters)):
        print('Houston, we have a problem')
        return None, None
    m = {}
    p = {}
    for machine, cluster in enumerate(m_clusters):
        m[machine] = cluster

    for part, cluster in enumerate(p_clusters):
        p[part] = cluster

    m = sorted(m.items(), key=lambda x: x[1])
    p = sorted(p.items(), key=lambda x: x[1])

    matrix = np.zeros((len(m) + 2, len(p) + 2), dtype=int)

    for i in range(len(p)):
        matrix[0][i + 2] = p[i][1]
        matrix[1][i + 2] = p[i][0]

    for i in range(len(m)):
        matrix[i + 2][0] = m[i][1]
        matrix[i + 2][1] = m[i][0]

    for i in range(len(m)):
        for j in range(len(p)):
            matrix[i + 2, j + 2] = ind_matrix[matrix[i + 2, 1], matrix[1, j + 2]]

    ones = matrix[2:, 2:].sum()
    max_clust = m_clusters.max() + 1

    clusters = []

    for clust in range(max_clust):
        ind_p = np.where(matrix[0, 2:] == clust)[0]
        ind_m = np.where(matrix.T[0, 2:] == clust)[0]
        clusters.append(matrix[ind_m[0] + 2:ind_m[-1] + 3, ind_p[0] + 2:ind_p[-1] + 3])

    n1_in = 0
    n0_in = 0

    for clust in clusters:
        n1_in += clust.sum()
        n0_in += (clust.shape[0] * clust.shape[1]) - clust.sum()

    eff = n1_in / (ones + n0_in)

    return matrix, eff

def get_res(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        data = data.split('\n')
        m_clust = data[0]
        p_clust = data[1]

        m_clust = np.asarray(m_clust.split(' '))
        del_ind = np.where(m_clust == '')[0].tolist()
        m_clust = np.delete(m_clust, del_ind)

        p_clust = np.asarray(p_clust.split(' '))
        del_ind = np.where(p_clust == '')[0].tolist()
        p_clust = np.delete(p_clust, del_ind)

        m = []
        p = []

        for m_c in m_clust:
            m.append(m_c.split('_')[1])

        for p_c in p_clust:
            p.append(p_c.split('_')[1])

    return np.asarray(m, dtype=int), np.asarray(p, dtype=int)



