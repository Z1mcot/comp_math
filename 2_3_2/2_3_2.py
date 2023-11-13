import numpy as np

def newton_raphson(f, J, x0, max_iter=100, tol=1e-6):
    x = x0
    for iter in range(max_iter):
        # Решаем систему J*dx = -f(x) для dx, где dx = x_(k+1) - x_k
        dx = np.linalg.solve(J(x), -f(x))

        # Стоит читать как x_(k+1) = x_k + dx
        x += dx
        
        print(f'Итерация {iter+1}: \n{x}\n')
        
        if np.linalg.norm(dx) < tol:
            return x, iter+1, True
    
    # Если не сходится
    return x, max_iter, False

# Система уравнений
def equations(x):
    f1 = x[0]**2 - x[1]**2 - 1
    f2 = x[0]*(x[1]**3) - x[1] - 1
    return np.array([f1, f2])

# Функция, вычисляющая матрицу Якоби
def jacobian(x):
    J = np.array([
        [2*x[0], -2*x[1]], 
        [x[1]**3, x[0]*3*(x[1]**2)]
    ])
    return J

# Начальное приближение
x0 = np.array([0.5, 0.5])

# Решаем систему уравнений
solution, iterations, converged = newton_raphson(equations, jacobian, x0)

# Выводим результат или сообщение о том, что метод не сходится
if (converged):
    print("Решение:", solution)
else:
    print('Решение не найденно. ' +
          'Попробуйте другое начальное приближение ' +
          'или повысьте количество итераций')

error = equations(solution)
print('Невязка: ', error)