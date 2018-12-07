import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.neighbors import KernelDensity

def sigmoid(x):
    return 1 / (1 + np.e**(-x))


def plot_classifier_performance(x_in, y_in, clf_output, generate, sess, pname='Clf_perf.pdf', batch=False):
        
    # predict on special values of Z
    n_samples = 10000
    X, Y, Z = generate(n_samples)
    X1, Y1, Z1 = generate(n_samples, z=1)
    X0, Y0, Z0 = generate(n_samples, z=0)
    X_1, Y_1, Z_1 = generate(n_samples, z=-1)
    pred = sigmoid(sess.run(clf_output, feed_dict={x_in:X, y_in:Y}))
    pred1 = sigmoid(sess.run(clf_output, feed_dict={x_in:X1, y_in:Y1}))
    pred0 = sigmoid(sess.run(clf_output, feed_dict={x_in:X0, y_in:Y0}))
    pred_1 = sigmoid(sess.run(clf_output, feed_dict={x_in:X_1, y_in:Y_1}))
    
    # define figure (plot grid)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    lim = 3
    
    # plot the variates
    N = 300
    ax[0,0].scatter(X[Y==0,0], X[Y==0,1], marker='o', color='k', alpha=0.2, label='Y=0')
    ax[0,0].scatter(X1[Y1==1,0][:N], X1[Y1==1,1][:N], marker='x', c='darkred', alpha=1, label='Y=1, Z=1')
    ax[0,0].scatter(X0[Y0==1,0][:N], X0[Y0==1,1][:N], marker='x', c='red', alpha=1, label='Y=1, Z=0')
    ax[0,0].scatter(X_1[Y_1==1,0][:N], X_1[Y_1==1,1][:N], marker='x', c='tomato', alpha=1, label='Y=1, Z=-1')
    ax[0,0].set_ylim(-lim, lim)
    ax[0,0].set_xlim(-lim, lim)
    ax[0,0].set_xlabel('x1')
    ax[0,0].set_ylabel('x2')
    ax[0,0].set_title('Variates')
    leg = ax[0,0].legend(loc='best')

    # plot the ROC curves for Z={-1,0,1}
    fpr1, tpr1, _ = roc_curve(Y1[Z1==1], pred1[Z1==1])
    fpr0, tpr0, _ = roc_curve(Y0[Z0==0], pred0[Z0==0])
    fpr_1, tpr_1, _ = roc_curve(Y_1[Z_1==-1], pred_1[Z_1==-1])
    ax[1,0].plot(fpr1, tpr1, c='darkred', label='Z=1')
    ax[1,0].plot(fpr0, tpr0, c='red', label='Z=0')
    ax[1,0].plot(fpr_1, tpr_1, c='tomato', label='Z=-1')
    ax[1,0].legend(loc='best')
    ax[1,0].set_xlabel('False positive rate')
    ax[1,0].set_ylabel('True positive rate')
    ax[1,0].set_title('ROC curve')
    
    # plot the probability densities for Z=-1,0,1
    bins=30
    ax[1,1].hist(pred1, bins=bins, density=True, color='darkred', histtype='step', label='Z=1')
    ax[1,1].hist(pred0, bins=bins, density=True, color='red', histtype='step', label='Z=0')
    ax[1,1].hist(pred_1, bins=bins, density=True, color='tomato', histtype='step', label='Z=-1')
    ax[1,1].set_xlabel('classifier output f(X|Z=z)')
    ax[1,1].set_ylabel('a.u.')
    ax[1,1].legend(loc='best')
    ax[1,1].set_title('Classifier output')
    
    # plot the decision boundary
    dec = ax[0,1].tricontourf(X[:,0], X[:,1], pred.ravel(), 20)
    ax[0,1].set_ylim(-lim, lim)
    ax[0,1].set_xlim(-lim, lim)
    ax[0,1].set_title('Decision boundary')
    ax[0,1].set_xlabel('x1')
    ax[0,1].set_ylabel('x2')
    plt.colorbar(dec, ax=ax[0,1])

    if not batch:
        plt.show()
    plt.savefig(pname)
    plt.close(fig)


def plot_Z_density(Z, fX, NLLs, n_adv_cycles, n_curves=3, pname='Z_density.pdf', batch=False):
    # make the cuts
    boundaries = np.linspace(0, 1, n_curves+1)
    upper = np.array(boundaries[1:])
    lower = np.array(boundaries[:-1])
    centres = (upper+lower)/2.
    
    # make the Z distributions (kernels) in cuts of f(X)
    Z_plot = np.linspace(-4, 4, 100)
    Zs = {}
    log_dens = {}
    
    for i in range(len(centres)):
        low = lower[i]
        upp = upper[i]
        Zs[i] = Z[np.logical_and(fX>low, fX<upp)]
    
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Zs[i].reshape(-1,1))
        log_dens[i] = kde.score_samples(Z_plot.reshape(-1,1))
    
    # make the plot
    fig, ax = plt.subplots(2, figsize=(7,7), gridspec_kw={'height_ratios':[1, 3]})
    plt.subplots_adjust(hspace=0.5)
    ax[0].plot(range(1, len(NLLs)+1), NLLs, c='k')
    ax[0].set_xlabel('Adversarial cycles')
    ax[0].set_ylabel('NLL')
    ax[0].set_xlim(1, n_adv_cycles)
    cmap = matplotlib.cm.get_cmap('Reds')
    for i in Zs:
        label = 'p( z | f(X)>{:1.2f} & f(X)<{:1.2f})'.format(lower[i], upper[i])
        ax[1].plot(Z_plot, np.exp(log_dens[i]), label=label, color=cmap(centres[i]))
    ax[1].set_xlabel('Sensitive parameter Z')
    ax[1].set_ylabel('p(z|f(X))')
    ax[1].set_xlim(-4,4)
    ax[1].set_ylim(0,0.7)
    ax[1].legend(loc='best')

    if not batch:
        plt.show()
    plt.savefig(pname)
    plt.close(fig)


def plot_toy_variates(X, Y, Z):
    n_samples = X.shape[0]
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(X[Y==0,0], X[Y==0,1], marker='o', color='k', alpha=0.2, label='Y=0')
    ax.scatter(X[Y==1,0], X[Y==1,1], marker='x', c=Z[n_samples//2:], alpha=0.4, cmap='Reds', label='Y=1')
    ax.set_ylim(-5, 5)
    ax.set_xlim(-5, 5)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    leg = ax.legend(loc='best')
    leg.legendHandles[1].set_color('red')
    fig.show()
