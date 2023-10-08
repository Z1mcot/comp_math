import json
import os
import numpy as np

def convert_to_float(arr):
    float_arr = np.array(arr, dtype=np.float64)
    return float_arr

def update_rows(row_i, row_j, c, s, n):
    return np.array([
        [c*row_i[q] + s*row_j[q] for q in range(n)],
        [-s*row_i[q] + c*row_j[q] for q in range(n)]
    ])

def rotate(A, F):
    A_n = A.shape[0]
    F_m = F.shape[1]
    
    X = np.zeros(shape=(A_n, F_m))

    # Прямой ход
    for i in range(A_n-1):
        for j in range(i+1, A_n):
            if A[j, i] != 0:
                c = A[i, i] / np.sqrt(A[i, i]**2 + A[j, i]**2)
                s = A[j, i] / np.sqrt(A[i, i]**2 + A[j, i]**2)

                A[[i, j]] = update_rows(A[i].copy(), A[j].copy(), c, s, A_n)
                
                F[[i, j]] = update_rows(F[i].copy(), F[j].copy(), c, s, F_m)

    # Обратный ход
    for k in range(A_n-1, -1, -1):
        prev = calc_prev_xs(X[k+1:], A[k, k+1:])
        X[k] = (F[k] - prev) / A[k, k]
    
    return X

def calc_prev_xs(X, A):
    if (X.shape[0] == 0):
        return 0
    
    res = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        res = res + np.array([elem * A[i] for elem in X[i]])
    
    return res

def calc_residual_matrix(A, X, F):
    return F - np.matmul(A, X) # где matmul - матричное умножение    

# Для красивого вывода ответа
def round_array(arr, digits = 2):
    rounded_arr = np.round(arr, digits)
    return rounded_arr

def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # Считываем данные из файла
    with open(f'{root_dir}/matrix.json', 'r') as file:
        data = json.load(file)

    # Приводим считанные матрицы к типу float
    A = convert_to_float(data['A'])
    F = convert_to_float(data['F'])

    # используме копии, для того, чтобы не модифицировать исходные A и F
    X = round_array(rotate(A.copy(), F.copy()))

    
    print("Решение:\n", X)

    residual = calc_residual_matrix(A, X, F)

    print("Матрица невязки:\n", round_array(residual, 5))

if __name__ == '__main__':
    main()

    