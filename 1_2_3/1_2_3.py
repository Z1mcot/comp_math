import json
import os
import numpy as np

# Функция, реализующая метод последовательных верхних релаксаций (метод Зейделя)
def sor_method(A, f, gamma=1.2, initial_guess=None, max_iterations=100, tolerance=1e-6):
    if initial_guess is None:
        x = np.zeros_like(f)
    else:
        x = initial_guess
    
    for _ in range(max_iterations):
        x_new = x.copy()
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (1 - gamma)*x[i] + (gamma / A[i, i])*(f[i] - s1 - s2)
        
        if np.allclose(x, x_new, atol=tolerance, rtol=0):
            return x_new
        
        x = x_new
    
    return x

def calc_residual_matrix(A, X, F):
    return F - np.matmul(A, X) # где matmul - матричное умножение    

def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    # Считываем данные из файла
    with open(f'{root_dir}/matrices.json', 'r') as file:
        data = json.load(file)

    A_matrices = data['A_matrices']
    f = np.array(data['f'])

    for i in range(len(A_matrices)):
        A = np.array(A_matrices[i])
        print(f"Для исходных данный под номерами {i+1} и 6:\n")

        X = sor_method(A.copy(), f.copy(), gamma=1.2, initial_guess=None, max_iterations=100, tolerance=1e-6)
        print("Решение:\n", X)

        residual = calc_residual_matrix(A, X, f)
        print("Матрица невязки:\n", residual)

if __name__ == '__main__':
    main()