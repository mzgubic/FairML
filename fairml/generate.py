import numpy as np

def generate_toys(n_samples, z=None):

    # Y
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # Z
    if z == None:
        #Z = np.random.normal(0, 1, size=n_samples)
        #Z = np.random.exponential(scale = 0.5, size = n_samples) - 2
        #Z = np.random.uniform(low = -2, high = 2, size = n_samples)
        Z = np.random.choice([-2, 0, 2], p = [0.33, 0.33, 0.34], size = n_samples)
        #Z = np.random.choice([-2, 0, 2], p = [0.5, 0.5], size = n_samples)
    else:
        Z = z * np.ones(n_samples)

    # X
    X0 = np.random.multivariate_normal([0, 0], [[1, -0.5],[-0.5, 1]], size=n_samples//2)
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z[n_samples//2:]
    X = np.concatenate([X0, X1])

    return X, Y, Z

def toys_single_Z_diff(n_samples, z=None):

    sigma = 1
    dx = 2

    # Y is the target
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # Z depends on Y (for training, can evaluate on any samples)
    if z == None:
        # simple gaussian: Z|(Y==0) ~ N(0, 1)
        Z0 = np.random.normal(0, sigma, size=n_samples//2)

        # comes from one of the two: Z|(Y==1) ~ N(-1, 1) + N(1, 1)
        Z1_1 = np.random.normal(-dx, sigma, size=n_samples//2)
        Z1_2 = np.random.normal(dx, sigma, size=n_samples//2)
        Z1_12 = np.concatenate([Z1_1, Z1_2]).reshape(2, -1)
        index = np.random.randint(0, dx, n_samples//2)
        Z1 = Z1_12[index, np.arange(len(index))]

        # combine them
        Z = np.concatenate([Z0, Z1])

    else:
        Z = z * np.ones(n_samples)

    # and Xs of course depend on both Z and Y
    X0 = np.random.multivariate_normal([0, 0], [[sigma, -.5*sigma], [-.5*sigma, sigma]], size=n_samples//2)
    X0[:, 1] += Z0
    X1 = np.random.multivariate_normal([dx, 0], [[sigma, 0], [0, sigma]], size=n_samples//2)
    X1[:, 1] += Z1
    X = np.concatenate([X0, X1])

    return X, Y, Z


