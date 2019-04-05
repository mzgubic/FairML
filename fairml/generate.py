import numpy as np

def toys_simple(n_samples, z=None):

    # Y
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # Z
    if z == None:
        Z0 = np.zeros(n_samples//2)
        Z1 = np.random.normal(0, 1, size=n_samples//2)
    else:
        Z0 = z * np.ones(n_samples//2)
        Z1 = z * np.ones(n_samples//2)
    Z = np.concatenate([Z0, Z1])

    # X
    X0 = np.random.multivariate_normal([0, 0], [[sigma, -.5*sigma], [-.5*sigma, sigma]], size=n_samples//2)
    X0[:,1] += Z0
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z1
    X = np.concatenate([X0, X1])

    return {'X':X, 'Y':Y.reshape(-1, 1), 'Z':Z.reshape(-1, 1)}


def toys_expo_Z(n_samples, z=None, sigma=1.0):

    # Y is the target
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # Z depends on Y (for training, can evaluate on any samples)
    if z == None:
        Z0 = np.random.exponential(1.0, size=n_samples//2)
        Z1 = np.random.exponential(2.0, size=n_samples//2)
    else:
        Z0 = z * np.ones(n_samples//2)
        Z1 = z * np.ones(n_samples//2)
    Z = np.concatenate([Z0, Z1])

    # and Xs of course depend on both Z and Y
    X0 = np.random.multivariate_normal([0, 0], [[sigma, -.5*sigma], [-.5*sigma, sigma]], size=n_samples//2)
    X0[:,1] += Z0
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z1
    X = np.concatenate([X0, X1])

    return {'X':X, 'Y':Y.reshape(-1, 1), 'Z':Z.reshape(-1, 1)}


def toys_discrete_Z(n_samples, z=None, sigma=1.0):

    # Y is the target
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    # Z depends on Y (for training, can evaluate on any samples)
    if z==None:
        Z0 = np.zeros(n_samples//2)
        Z1 = np.random.choice([-2, 0, 2], p=[0.33, 0.33, 0.34], size=n_samples//2)
    else:
        Z0 = z * np.ones(n_samples//2)
        Z1 = z * np.ones(n_samples//2)
    Z = np.concatenate([Z0, Z1])

    # and Xs of course depend on both Z and Y
    X0 = np.random.multivariate_normal([0, 0], [[sigma, -.5*sigma], [-.5*sigma, sigma]], size=n_samples//2)
    X0[:,1] += Z0
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z1
    X = np.concatenate([X0, X1])

    return {'X':X, 'Y':Y.reshape(-1, 1), 'Z':Z.reshape(-1, 1)}

