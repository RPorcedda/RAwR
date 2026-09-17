import numpy as np


def generate_data_from_S(
    S,
    n_ch=3,
    d_in=10,
    d_out=7,
    ret_all=False,
    seed=123434,
    sigma_arr=None,
    X=None,
):
    np.random.seed(seed)
    N = S.shape[0]
    if X is None:
        X = np.random.normal(0.0, 1.0, (N, d_in))
    else:
        X = np.asarray(X, dtype=float)
        if X.shape[0] != N:
            raise ValueError(f"X has {X.shape[0]} rows but S is {N}x{N}")
        d_in = X.shape[1]

    if sigma_arr is None:
        sigma_arr = [1.0] * n_ch
    if len(sigma_arr) != n_ch:
        raise ValueError(f"sigma_arr must have length n_ch={n_ch}")

    W_arr = []
    for i in range(n_ch):
        sigma = float(sigma_arr[i])
        W_arr.append(np.random.normal(0.0, sigma / np.sqrt(d_in), (d_out, d_in)))
    S_arr = [np.linalg.matrix_power(S, i) for i in range(n_ch)]

    Y_partial = [S_arr[i] @ X @ W_arr[i].T for i in range(n_ch)]
    Y = sum(Y_partial)
    if ret_all:
        return X, Y, S_arr, W_arr
    return X, Y
