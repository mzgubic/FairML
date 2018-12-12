import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def generate_toys(n_samples, z=None):
    """
    Convenience function which generates toy examples on the fly.

    Args:
        n_samples (int): Number of samples to be generated
        z (float): Value of the nuisance parameter to be used, None gives a normal distro of Z values.

    Returns:
        X, Y, Z: features, target, sensitive variable.
    """
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


def generate_hmumu(features='low'):
    """
    Get a convenience function which generates hmumu events.

    Args:
        features (string): Which features are used in the dataset. Can be 'low', 'high', or 'both'.

    Returns:
        x_scaler (StandardScaler): Object which scaled the X data
        z_scaler (StandardScaler): Object which scaled the Z data
        generate (function): Function which samples instances of the data
    """

    # first, load the dataset
    df = pd.read_csv('../data/Combined_10000.csv')
    n_tot = df.shape[0]
    n_sig = np.sum(df.IsSignal == 1)
    n_bkg = np.sum(df.IsSignal == 0)
    print('------------------------')
    print('{} signal events.'.format(n_sig))
    print('{} background events.'.format(n_bkg))
    print('------------------------')

    # create the X, Y, Z, W frames
    def create_frames(df, features):

        low_level = ['Muons_Eta_Lead', 'Muons_Eta_Sub', 'Muons_Phi_Lead', 'Muons_Phi_Sub', 'Muons_PT_Lead', 'Muons_PT_Sub']
        high_level = ['Z_PT', 'Muons_CosThetaStar']

        if features == 'low':
            X_names = low_level
        elif features == 'high':
            X_names = high_level
        elif features == 'both':
            X_names = low_level+high_level

        Z_names = ['Muons_Minv_MuMu']
        Y_names = ['IsSignal']
        W_names = ['GlobalWeight']

        X = df[X_names]
        Z = df[Z_names]
        Y = df[Y_names].values
        W = df[W_names].values

        return X, Y, Z, W

    X, Y, Z, W = create_frames(df, features)

    # normalise all features
    x_scaler = StandardScaler()
    z_scaler = StandardScaler()
    scaled_X = x_scaler.fit_transform(X)
    scaled_Z = z_scaler.fit_transform(Z)

    # define a generator which randomly samples the data
    def generator(X, Y, Z, W, balanced=False):
        """
        Create a generator sampling from inputs datasets.

        Args:
            X, Y, Z, W (arrays): features, targets, sensitive attributes, weights.
            balanced (bool): sample balanced or unbalanced (sig vs bkg) events.

        Yields:
            X, Y, Z, W (arrays): samples from the input datasets.
        """

        while True:

            n_samples = yield # feed how many samples you'd like in each iteration (using send method)

            # balanced classes (training)
            # ----------------
            # - same number of sig/bkg classes
            # - equal total weights for both
            if balanced:

                # create separate sig and bkg frames
                sig_ind = (Y == 1).reshape(-1)
                n_sig = np.sum(sig_ind)
                X_sig = X[sig_ind, :]
            
                Y_sig = Y[sig_ind, :]
                Z_sig = Z[sig_ind, :]
                W_sig = W[sig_ind, :]
            
                bkg_ind = (Y == 0).reshape(-1)
                n_bkg = np.sum(bkg_ind)
                X_bkg = X[bkg_ind, :]
                Y_bkg = Y[bkg_ind, :]
                Z_bkg = Z[bkg_ind, :]
                W_bkg = W[bkg_ind, :]

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
                n_tot = X.shape[0]
                indices = np.random.randint(0, n_tot, size=n_samples)
                yield X[indices, :], Y[indices, :], Z[indices, :], W[indices, :]

    # create generator instances (called in the convenience function later on)
    X_tr, X_te, Y_tr, Y_te, Z_tr, Z_te, W_tr, W_te = train_test_split(scaled_X, Y, scaled_Z, W, random_state=42, test_size=0.2)

    balanced_train   = generator(X_tr, Y_tr, Z_tr, W_tr, balanced=True)
    imbalanced_train = generator(X_tr, Y_tr, Z_tr, W_tr, balanced=False)
    balanced_test    = generator(X_te, Y_te, Z_te, W_te, balanced=True)
    imbalanced_test  = generator(X_te, Y_te, Z_te, W_te, balanced=False)

    # and return the convenience function which calls the correct instance of the generator
    def generate(n_samples, train=True, balanced=True):
        """
        Sample hmumu events with replacement.

        Args:
            n_samples (int): how many events to sample.
            train (bool): sample train or test part of the dataset.
            balanced (bool): sample balanced or unbalanced (sig vs bkg) events.

        Returns:
            X, Y, Z, W (arrays): features, targets, sensitive attributes, weights.
        """

        # choose the correct generator
        if balanced and train:
            g = balanced_train

        elif not balanced and train:
            g = imbalanced_train

        elif balanced and not train:
            print('Go home, you are drunk. Do you really want to test on balanced dataset?')
            g = balanced_test

        elif not balanced and not train:
            g = imbalanced_test
            
        # get the samples from the correct generator
        next(g)
        X, Y, Z, W = g.send(n_samples)
        print('Getting {} examples.'.format(X.shape[0]))
        return X, Y.reshape(-1), Z.reshape(-1), W.reshape(-1)

    return x_scaler, z_scaler, generate


