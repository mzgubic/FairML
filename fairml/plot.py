import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.neighbors import KernelDensity

x_min, x_max = -3, 3
bandwidth = 0.2

def XYZ(batch):

    X, Y, Z = batch['X'], batch['Y'], batch['Z']
    Y = Y.ravel()
    Z = Z.ravel()
    
    return X, Y, Z


def variates_main(ax, batch):
    
    # prepare
    X, Y, Z = XYZ(batch)
    n_samples = X.shape[0]
    
    # plot
    ax.scatter(X[Y==0,0], X[Y==0,1], marker='o', color='k', alpha=0.2, label='Y=0')
    ax.scatter(X[Y==1,0], X[Y==1,1], marker='x', c=Z[n_samples//2:], alpha=0.4, cmap='Reds', label='Y=1')
    
    # cosmetics
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    leg = ax.legend(loc='best')
    leg.legendHandles[1].set_color('red')
    

def variates_kde(ax, batch, x_cpt):
    
    # prepare
    X, Y, Z = XYZ(batch)
    
    # gaussian KDE
    x_plot = np.linspace(x_min, x_max, 100)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    log_0 = kde.fit(X[Y==0,x_cpt].reshape(-1,1)).score_samples(x_plot.reshape(-1,1))
    log_1 = kde.fit(X[Y==1,x_cpt].reshape(-1,1)).score_samples(x_plot.reshape(-1,1))
    
    # top
    if x_cpt == 0:
        ax.plot(x_plot, np.exp(log_0), color='k')
        ax.plot(x_plot, np.exp(log_1), color='r')
        
    # right (change x, y in plotting to make a vertical plot)
    else:
        ax.plot(np.exp(log_0), x_plot, color='k')
        ax.plot(np.exp(log_1), x_plot, color='r')
        

def roc_curves(ax, batch1, batch0, batch_1, preds1, preds0, preds_1):
    
    # prepare
    X1, Y1, Z1 = XYZ(batch1)
    X0, Y0, Z0 = XYZ(batch0)
    X_1, Y_1, Z_1 = XYZ(batch_1)
    
    # plot the ROC curves for Z={-1,0,1}
    fpr1, tpr1, _ = roc_curve(Y1, preds1)
    fpr0, tpr0, _ = roc_curve(Y0, preds0)
    fpr_1, tpr_1, _ = roc_curve(Y_1, preds_1)
    ax.plot(fpr1, tpr1, c='darkred', label='Z=1')
    ax.plot(fpr0, tpr0, c='red', label='Z=0')
    ax.plot(fpr_1, tpr_1, c='tomato', label='Z=-1')
    ax.legend(loc='best')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('ROC curve')
    

def clf_outputs(ax, preds1, preds0, preds_1):
    
    bins=30
    ax.hist(preds1, bins=bins, density=True, color='darkred', histtype='step', label='Z=1')
    ax.hist(preds0, bins=bins, density=True, color='red', histtype='step', label='Z=0')
    ax.hist(preds_1, bins=bins, density=True, color='tomato', histtype='step', label='Z=-1')
    ax.set_xlabel('classifier output f(X|Z=z)')
    ax.set_ylabel('a.u.')
    ax.legend(loc='best')
    ax.set_title('Classifier output')
    

def decision_boundary(ax, batch, preds):
    
    # prepare
    X, Y, Z = XYZ(batch)
    
    # plot
    dec = ax.tricontourf(X[:,0], X[:,1], preds.ravel(), 20)
    ax.set_ylim(x_min, x_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title('Decision boundary')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.colorbar(dec, ax=ax)


def history(ax, metric, style='-', color='k', label='No label', cut_first=0, show_individual=False):

    # in case multiple values are given, compute mean and std and show them
    do_uncertainty = isinstance(metric[0], list) or isinstance(metric[0], np.ndarray)
    
    # compute mean and std
    if do_uncertainty:
        n_histories = len(metric)
        n_steps = len(metric[0])
        histories = np.zeros(shape=(n_histories, n_steps))
        for i, h in enumerate(metric):
            histories[i] = np.array(metric[i])
            
        #middle = np.mean(histories, axis=0)
        #up = middle+np.std(histories, axis=0)
        #down = middle-np.std(histories, axis=0)
        middle = np.percentile(histories, 50, axis=0)
        up = np.percentile(histories, 84, axis=0)
        down = np.percentile(histories, 16, axis=0)
    else:
        n_steps = len(metric)
        middle = np.array(metric)
    
    # cut away first N to get 'zoom in' effect on the y-scale
    n_cut = cut_first if n_steps > 30 else 0

    # plot
    xs = range(n_cut, n_steps)
    if do_uncertainty:

        # plot individual losses
        if show_individual:
            for i in range(n_histories):
                ax.plot(xs, histories[i][n_cut:], linestyle=style, c=color, alpha=0.2)

        # plot std dev
        ax.fill_between(xs, down[n_cut:], up[n_cut:], color=color, alpha=0.2)
        
    # in any case plot the mean
    ax.plot(xs, middle[n_cut:], linestyle=style, c=color, label=label)
    ax.set_xlim(0, n_steps)
