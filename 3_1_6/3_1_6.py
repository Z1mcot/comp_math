import numpy as np

# Интегрируемая функция
def f(x):
    return np.exp(x) 

def newton_cotes(f, a, b, n, B_k, B_common_denomenator):
    h = (b - a) / n
    
    area = (b - a) / B_common_denomenator * np.sum([B_k[k] * f(a + k * h) for k in range(n+1)])
    return area

a = 0.0  # Нижняя граница интегрирования
b = 1.0  # Верхняя граница интегрирования
n = 6  # Количество подинтервалов
B_k = [41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0] # B-коэффициенты без знаменателя
B_common_denomenator = 840.0 # общий знаменатель всех B-коэффициентов

result = newton_cotes(f, a, b, n, B_k, B_common_denomenator)
print("Результат интегрирования:", result)
