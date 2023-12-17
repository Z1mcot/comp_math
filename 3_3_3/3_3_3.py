import json
import os
import numpy as np
from sympy import symbols

_factories = {
    str: lambda x: symbols(x) if '+' not in x else plus_sign_handler(x),
    int: lambda x: float(x),
    float: lambda x: x
}

def plus_sign_handler(x: str):
    val = x.split(' ')
    return symbols(val[0]) + symbols(val[2])

def parse_json(A):
    res = []
    for i, v in enumerate(A):
        res.append([_factories[type(j)](j) for j in v])
    return np.array(res)

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    try:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    except:
        return False

# Функция для выполнения циклического метода Якоби
def jacobi_iteration(A, num_iterations=100, tolerance=1e-6):
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    
    for iteration in range(num_iterations):
        off_diagonal_sum = np.sum(np.abs(A - np.diag(np.diagonal(A))))
        
        
        if off_diagonal_sum < tolerance:
            break

        for i in range(n):
            for j in range(i + 1, n):
                if np.abs(A[i, j]) >= tolerance:
                    if A[i,i] != A[j,j]:
                        theta = (A[j, j] - A[i, i]) / (2 * A[i, j])
                        t = np.sign(theta) / (np.abs(theta) + np.sqrt(1 + theta**2))
                        c = 1.0 / np.sqrt(1 + t**2)
                        s = c * t
                    else: 
                        theta = np.pi * 0.25
                        c = np.cos(theta)
                        s = np.sin(theta)

                    old_ii = A[i, i]
                    old_jj = A[j, j]
                    A[i, i] = c**2 * old_ii - 2 * c * s * A[i, j] + s**2 * old_jj
                    A[j, j] = s**2 * old_ii + 2 * c * s * A[i, j] + c**2 * old_jj
                    A[i, j] = A[j, i] = 0

                    for k in range(n):
                        if k != i and k != j:
                            old_ik = A[i, k]
                            old_jk = A[j, k]
                            A[i, k] = c * old_ik - s * old_jk
                            A[k, i] = A[i, k]
                            A[j, k] = s * old_ik + c * old_jk
                            A[k, j] = A[j, k]

    eigenvalues = np.diagonal(A)
    return eigenvalues

def calculate_eigenvectors(A, eigenvalues):
    _, eigenvectors = np.linalg.eig(A)

    return eigenvectors

# def round_array(arr, digits = 6):
#     rounded_arr = np.round(arr, digits)
#     return rounded_arr

def calc_residual_matrix(matrix, eigenvalues, eigenvectors):
    result = []
    n = matrix.shape[0]
    for i in range(n):
        result.append(matrix*eigenvectors[i] - eigenvalues[i]*eigenvectors[i])
    return result

def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # Считываем данные из файла
    with open(f'{root_dir}/matrices.json', 'r') as file:
        data = json.load(file)

    A_matrices = data['A_matrices']

    for jsonMatrix in A_matrices:
        matrix = parse_json(jsonMatrix)
        print(f'\nДля матрицы:\n{matrix}')
        if (check_symmetric(matrix) is False):
            print('Метод не работает с несимметричными матрицами')
            continue

        eigenvalues = jacobi_iteration(matrix.copy(), num_iterations=100, tolerance=1e-6)
        print("Собственные значения:", eigenvalues)

        eigenvectors = calculate_eigenvectors(matrix.copy(), eigenvalues)
        print("Собственные векторы:\n", eigenvectors)

        residual = calc_residual_matrix(matrix.copy(), eigenvalues, eigenvectors)

        for i in range(matrix.shape[0]):
            print(f'Матрица невязки для собственного числа {eigenvalues[i]} и собственного вектора ' +
                  f'{eigenvectors[i]}:\n\n', np.array(residual[i]))

if __name__ == '__main__':
    main()