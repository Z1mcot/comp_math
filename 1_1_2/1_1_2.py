import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Создает СЛАУ с базисными функциямми
def create_matrix(X_vector):
    n = len(X_vector)
    matrix = np.ones((n, n))  # Заполняем первый столбец единицами

    for i in range(n):
        for j in range(1, n, 2):
            matrix[i, j] = np.exp(X_vector[i] * ((j + 1) // 2))
            matrix[i, j + 1] = np.exp(-X_vector[i] * ((j + 1) // 2))
                
    return matrix

def solve_coefficients(A, f):
    return np.linalg.solve(A, f)

# Исходная функция
def f(x):
    return 0.25*(np.exp(x*2) ** 2) + x**4 

def g(coefficients, x):
    n = len(coefficients)
    a_0 = coefficients[0]
    sum_body = []
    
    for i in range(1, n, 2):
        sum_body.append(coefficients[i] * np.exp((i + 1) // 2 * x) + coefficients[i+1] * np.exp(-(i + 1) // 2 * x))

    return a_0 + np.sum(sum_body)

def main():
    # Исходные данные
    X_vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    Y_vector = np.array([f(x) for x in X_vector])

    # Получаем матрицу с базисными функциями
    result_matrix = create_matrix(X_vector.copy())
    print('result_matrix: ', result_matrix)

    # Находим неизвестные коэффициенты
    coefficients = solve_coefficients(result_matrix, Y_vector.copy())
    print(coefficients)

    # Находим интерполированные значения в переделах (a, b)
    interpolated_X = [float(x) / 10 for x in range(X_vector.min() * 10, X_vector.max() * 10 + 1)]
    interpolated_Y = [g(coefficients, x) for x in interpolated_X]

    lib_interpolation = interp1d(X_vector, Y_vector, kind='cubic')

    plt.scatter(X_vector, Y_vector, label='Исходные данные')
    plt.plot(interpolated_X, interpolated_Y, 'r-', label='Наша интерполяция')
    plt.plot(interpolated_X, lib_interpolation(interpolated_X), 'g-', label='Библиотечная интерполяция')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()    