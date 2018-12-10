import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def generate_toys(n_samples, z=None):
    Y = np.zeros(n_samples)
    Y[n_samples//2:] = 1

    if z == None:
        Z = np.random.normal(0, 1, size=n_samples)
    elif z == '2gauss':
        Z0 = np.random.normal(0, 1, size=n_samples//2)
        Z1 = np.random.normal(1, 1, size=n_samples//2)
        Z = np.concatenate([Z0, Z1])
    else:
        Z = z * np.ones(n_samples)

    X0 = np.random.multivariate_normal([0, 0], [[1, -0.5],[-0.5, 1]], size=n_samples//2)
    X1 = np.random.multivariate_normal([1, 1], 0.5*np.eye(2), size=n_samples//2)
    X1[:,1] += Z[n_samples//2:]
    X = np.concatenate([X0, X1])

    # reshuffle to mix y=0,1
    x1, x2 = X[:,0], X[:,1]
    s = np.stack([x1, x2, Y, Z], axis=1)
    X = s[:, :2]
    Y = s[:, 2]
    Z = s[:, 3]

    return X, Y, Z


def generate_hmumu():

    # first, load the dataset
    df = pd.read_csv('../data/combined_100000.csv')
    n_total = df.shape[0]
    n_sig = np.sum(df.IsSignal == 1)
    n_bkg = np.sum(df.IsSignal == 0)

    # create the X, Y, Z, and W frames
    X_names = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Z_PT']
    Z_names = ['Muons_Minv_MuMu']
    Y_names = ['IsSignal']
    W_names = ['GlobalWeight']

    X = df[X_names]
    Z = df[Z_names]
    Y = df[Y_names].values
    W = df[W_names].values

    # normalise all features
    x_scaler = StandardScaler()
    z_scaler = StandardScaler()
    scaled_X = x_scaler.fit_transform(X)
    scaled_Z = z_scaler.fit_transform(Z)

    # create separate sig and bkg frames (for balancing)
    sig_ind = (Y == 1).reshape(-1)
    X_sig = scaled_X[sig_ind, :]

    Y_sig = Y[sig_ind, :]
    Z_sig = scaled_Z[sig_ind, :]
    W_sig = W[sig_ind, :]

    bkg_ind = (Y == 0).reshape(-1)
    X_bkg = scaled_X[bkg_ind, :]
    Y_bkg = Y[bkg_ind, :]
    Z_bkg = scaled_Z[bkg_ind, :]
    W_bkg = W[bkg_ind, :]

    # define a generator which randomly samples the data
    def generator(balanced=False):
        while True:

            n_samples = yield # feed how many samples you'd like in each iteration (using send method)

            # balanced classes (training)
            # ----------------
            # - same number of sig/bkg classes
            # - equal total weights for both
            if balanced:

                # sample sig events
                indices = np.random.randint(0, n_sig, size=n_samples//2)
                xs, ys, zs, ws = X_sig[indices, :], Y_sig[indices, :], Z_sig[indices, :], W_sig[indices, :]

                # sample bkg events
                indices = np.random.randint(0, n_bkg, size=n_samples//2)
                xb, yb, zb, wb = X_bkg[indices, :], Y_bkg[indices, :], Z_bkg[indices, :], W_bkg[indices, :]

                # normalise the weights
                ws_tot = np.sum(ws)
                wb_tot = np.sum(wb)
                ws = ws * (1.0/ws_tot)
                wb = wb * (1.0/wb_tot)

                yield np.vstack([xs,xb]), np.vstack([ys,yb]), np.vstack([zs,zb]), np.vstack([ws,wb])

            # imbalanced classes (evaluation)
            # ----------------
            # - do not change the underlying class distributions
            # - use Global (physical) weights
            else:

                # sample mixed events
                indices = np.random.randint(0, n_total, size=n_samples)
                yield scaled_X[indices, :], Y[indices, :], scaled_Z[indices, :], W[indices, :]

    # create a generator instance (called in the convenience function later on)
    balanced_gen_inst = generator(balanced=True)
    imbalanced_gen_inst = generator(balanced=False)

    # and return the convenience function which calls that instance of a generator to yield the n_samples
    def generate(n_samples, balanced=True):

        # balanced classes needed for training
        if balanced:
            next(balanced_gen_inst)
            X, Y, Z, W = balanced_gen_inst.send(n_samples)
            
        # imbalanced (original) needed for evaluation
        else:
            next(imbalanced_gen_inst)
            X, Y, Z, W = imbalanced_gen_inst.send(n_samples)

        # and return the correct shapes 
        return X, Y.reshape(-1), Z.reshape(-1), W.reshape(-1)

    return x_scaler, z_scaler, generate


