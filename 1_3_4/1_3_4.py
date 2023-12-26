import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Создает СЛАУ с базисными функциямми
def create_matrix(X_vector, N, n):
    
    matrix = np.zeros((N, n))  # Заполняем первый столбец единицами

    for i in range(N):
        for j in range(n):
            matrix[i, j] = np.cosh(X_vector[i] * j)
                
    return matrix

def solve_coefficients(A: np.ndarray, f , N, n):
    matrix = np.zeros((n, n))
    right_side = np.zeros(n)
    for i in range(n):
        for k in range(n):
            matrix[i, k] = np.sum([A[j, k] * A[j, i] for j in range(N)])
            right_side[i] = np.sum([f[j] * A[j, i] for j in range(N)])

    return np.linalg.solve(matrix, right_side)


# Исходная функция
def f(x):
    return np.sin(x)

def g(coefficients, x, n):
    sum_body = [coefficients[k] * np.cosh(k * x) for k in range(n)]
    return np.sum(sum_body)

def main():
    # Исходные данные
    X_vector = np.array(np.linspace(0, 3, 10))
    Y_vector = np.array([f(x) for x in X_vector])

    N = len(X_vector)
    n = N // 2

    # Получаем матрицу с базисными функциями
    result_matrix = create_matrix(X_vector.copy(), N, n)

    # Находим неизвестные коэффициенты
    coefficients = solve_coefficients(result_matrix, Y_vector.copy(), N, n)
    print(coefficients)

    # Находим интерполированные значения в переделах (a, b)
    interpolated_X = np.linspace(X_vector.min(), X_vector.max())
    interpolated_Y = [g(coefficients, x, n) for x in interpolated_X]

    plt.scatter(X_vector, Y_vector, label='Исходные данные')
    plt.plot(interpolated_X, interpolated_Y, 'r-', label='Наша аппроксимация')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()    