from os.path import join
import numpy as np

def get_data(file_name):

    case_dict = {}
    case_dict['file_name'] = file_name
    path = './data/cfp_data'

    with open(join(path, file_name), 'r') as file:
        data = file.read()
        data = data.split('\n')
        sizes = data.pop(0).split(' ')
        case_dict['m'] = int(sizes[0])
        case_dict['p'] = int(sizes[1])
        adj_matrix = np.zeros((case_dict['m'], case_dict['p']))
        for m, line in enumerate(data):
            line = np.asarray(line.split(' '))
            del_ind = np.where(line == '')[0].tolist()
            line = np.delete(line, [0] + del_ind)

            for p in line:
                adj_matrix[m, int(p)-1] = 1
        case_dict['adj_matrix'] = adj_matrix

        print(file_name, ' is done!')

    return case_dict
