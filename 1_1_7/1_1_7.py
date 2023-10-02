import json
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
    
    x = np.zeros(shape=(A_n, F_m))
    X = np.zeros(shape=(A_n, F_m))

    for i in range(A_n-1):
        for j in range(i+1, A_n):
            if A[j, i] != 0:
                c = A[i, i] / np.sqrt(A[i, i]**2 + A[j, i]**2)
                s = A[j, i] / np.sqrt(A[i, i]**2 + A[j, i]**2)

                A[[i, j]] = update_rows(A[i].copy(), A[j].copy(), c, s, A_n)
                # A[j] = np.array([-s*row_i[q] + c*row_j[q] for q in range(n)], dtype=float)
                
                F[[i, j]] = update_rows(F[i].copy(), F[j].copy(), c, s, F_m)

    for k in range(A_n-1, -1, -1):
        prev = calc_prev_xs(X[k+1:], A[k, k+1:])
        X[k] = np.divide(np.subtract(F[k], prev), A[k,k])
    
    return X

def calc_prev_xs(X, A):
    if (X.shape[0] == 0):
        return 0
    
    res = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        huy = np.array([elem * A[i] for elem in X[i]])
        res = res + np.array([elem * A[i] for elem in X[i]])
    return res

        

def round_to_two_decimal_places(arr, digits = 2):
    rounded_arr = np.round(arr, digits)
    return rounded_arr

def main():
    with open('./matrix.json', 'r') as file:
        data = json.load(file)

    A = convert_to_float(data['A'])
    F = convert_to_float(data['F'])

    x = rotate(A, F)

    print("Solution:", *x)

if __name__ == '__main__':
    main()

# A = np.array([[1, 2, 3],
#               [3, 5, 7],
#               [1, 3, 4]])

# f = np.array([3, 0, 1])

# A_float = convert_to_float(A)
# f_float = convert_to_float(f)


