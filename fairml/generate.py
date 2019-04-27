import numpy as np

def generate_toys(n_samples, z=None):

    X = []
    Y = []
    Z = []
    
    for samp in range(n_samples):
        if z is None:
            # play a small trick here: train on a slightly larger range of the sensitive
            # parameter than what is used later for evaluation (use -1, 0, 1)
            z_cur = np.random.uniform(low = -1.1, high = 1.1, size = 1)[0]
        else:
            z_cur = z
        Z.append(z_cur)
        
        if np.random.rand() > 0.5:
            x = np.random.normal(-0.2, 0.5, size = 1)[0]
            y = 0
            X.append(x)
            Y.append(y)
        else:
            # map from z = (-1, 1) to nu = (0 + eps, 1 - eps)
            nu = z_cur / 3.0 + 0.5
            mu = 0.5 + nu
            var = 0.5 * (1 - nu)
            x = np.random.normal(mu, var, size = 1)[0]
            y = 1
            X.append(x)
            Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    X = np.expand_dims(X, axis = 1)
    #Y = np.expand_dims(Y, axis = 1)
    #Z = np.expand_dims(Z, axis = 1)
    
    # print("dim(X) = {}".format(np.shape(X)))
    # print("dim(Y) = {}".format(np.shape(Y)))
    # print("dim(Z) = {}".format(np.shape(Z)))
    
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


