import numpy as np
import matplotlib.pyplot as plt

# Создает СЛАУ с базисными функциямми
def create_matrix(X_vector):
    n = len(X_vector)
    matrix = np.ones((n, n))  # Заполняем первый столбец единицами

    for i in range(n):
        for j in range(1, n, 2):
            matrix[i, j] = np.exp(X_vector[i] * ((j + 1) // 2))
            matrix[i, j + 1] = np.exp(-X_vector[i] * ((j + 1) // 2))
                
    return matrix

def solve_equation(A, f):
    return np.linalg.solve(A, f)

# Исходная функция
def f(x):
    return np.exp(x) ** 2 + 4 * np.exp(x**2) - 10

def g(coefficients, x):
    n = len(X_vector)
    a_0 = coefficients[0]
    sum_body = []
    
    for i in range(1, n, 2):
        sum_body.append(coefficients[i] * np.exp((i + 1) // 2 * x) + coefficients[i+1] * np.exp(-(i + 1) // 2 * x))

    return a_0 + np.sum(sum_body)

# Example usage
X_vector = np.array([4, 5, 6, 7, 8, 10 ,12])
Y_vector = np.array([f(x) for x in X_vector])
result_matrix = create_matrix(X_vector.copy())
print('result_matrix: ', result_matrix)
coefficients = solve_equation(result_matrix, Y_vector.copy())
print(coefficients)

interpolated_X = [float(x) / 10 for x in range(40, 121)]
interpolated_Y = [g(coefficients, x) for x in interpolated_X]



plt.scatter(X_vector, Y_vector, label='Original data')
plt.plot(interpolated_X, interpolated_Y, 'r-', label='Fitted curve')
plt.legend()
plt.show()
