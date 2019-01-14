import os
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.neighbors import KernelDensity
import utils


def plot_classifier_performance(data, pname, batch=False):

    # predict on special values of Z
    n_samples = 10000
    X, Y, Z = data['all Z']
    X1, Y1, Z1 = data['Z=1']
    X0, Y0, Z0 = data['Z=0'] 
    X_1, Y_1, Z_1 = data['Z=-1']
    pred, pred1, pred0, pred_1 = data['preds']
    
    # define figure (plot grid)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(os.path.basename(pname).split('.')[0])
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

    plt.savefig(pname)
    if not batch:
        plt.show()
    plt.close(fig)


def plot_2D(fX, Y, Z, pname, batch=False):
    
    # percentiles
    Z_median = np.percentile(Z, 50)
    fX_median = np.percentile(fX, 50)
    
    # gaussian kernel density estimation
    Z_plot = np.linspace(-4, 4, 100)
    Z_log_dens = {}
    Z_kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    Z_log_dens['full'] = Z_kde.fit(Z.reshape(-1,1)).score_samples(Z_plot.reshape(-1,1))
    Z_log_dens['low'] = Z_kde.fit(Z[(fX<fX_median).reshape(-1)].reshape(-1,1)).score_samples(Z_plot.reshape(-1,1))
    Z_log_dens['high'] = Z_kde.fit(Z[(fX>fX_median).reshape(-1)].reshape(-1,1)).score_samples(Z_plot.reshape(-1,1))
    
    fX_plot = np.linspace(0, 1, 100)
    fX_log_dens = {}
    fX_kde = KernelDensity(kernel='gaussian', bandwidth=0.01)
    fX_log_dens['full'] = fX_kde.fit(fX.reshape(-1,1)).score_samples(fX_plot.reshape(-1,1))
    fX_log_dens['low'] = fX_kde.fit(fX[Z<Z_median].reshape(-1,1)).score_samples(fX_plot.reshape(-1,1))
    fX_log_dens['high'] = fX_kde.fit(fX[Z>Z_median].reshape(-1,1)).score_samples(fX_plot.reshape(-1,1))
    
    # plot
    fig, ax = plt.subplots(2, 2, figsize=(7,7), gridspec_kw={'height_ratios':[1,4], 'width_ratios':[4,1]}, sharex='col', sharey='row')
    fig.suptitle(os.path.basename(pname).split('.')[0])
    
    # fill and median line
    ax[1,0].fill_between([-4,Z_median], [0,0], [1,1], facecolor='tomato', interpolate=True, alpha=0.5)
    ax[1,0].fill_between([Z_median,4], [0,0], [1,1], facecolor='darkred', interpolate=True, alpha=0.5)
    ax[1,0].plot([-4, 4], [fX_median, fX_median], 'k:')

    # scatter
    indices = np.arange(0, len(Z), 500)
    Z_sc, Y_sc, fX_sc = Z[indices], Y[indices], fX[indices]
    col = np.where(fX_sc[Y_sc==0]<fX_median, 'royalblue', 'darkblue').reshape(-1)
    ax[1,0].scatter(Z_sc[Y_sc==0], fX_sc[Y_sc==0], c=col, marker='.')
    col = np.where(fX_sc[Y_sc==1]<fX_median, 'royalblue', 'darkblue').reshape(-1)
    ax[1,0].scatter(Z_sc[Y_sc==1], fX_sc[Y_sc==1], c=col, marker='x')
    
    ax[1,0].set_xlim(-4, 4)
    ax[1,0].set_ylim(0, 1)
    ax[1,0].set_xlabel('Sensitive parameter Z')
    ax[1,0].set_ylabel('f(X)')
    
    # top
    ax[0,0].set_ylabel('p(z)')
    ax[0,0].plot(Z_plot, np.exp(Z_log_dens['full']), color='k')
    ax[0,0].plot(Z_plot, 0.5*np.exp(Z_log_dens['low']), color='royalblue')
    ax[0,0].plot(Z_plot, 0.5*np.exp(Z_log_dens['high']), color='darkblue')
    
    # right
    ax[1,1].set_xlabel('p(f(X))')
    ax[1,1].plot(np.exp(fX_log_dens['full']), fX_plot, color='k')
    ax[1,1].plot(0.5*np.exp(fX_log_dens['low']), fX_plot, color='tomato')
    ax[1,1].plot(0.5*np.exp(fX_log_dens['high']), fX_plot, color='darkred')
    
    # empty
    fig.delaxes(ax[0,1])
    
    plt.savefig(pname)
    if not batch:
        plt.show()
    plt.close(fig)


def plot_losses(loss_D, loss_R, loss_DR, MIs, adversary, pname):

    # cut away first N to get 'zoom in' effect
    n_tot = len(loss_D)
    n_cut = 15 if len(MIs) > 30 else 0
    loss_D = loss_D[n_cut:]
    loss_R = loss_R[n_cut:]
    loss_DR = loss_DR[n_cut:]
    MIs = MIs[n_cut:]

    # plot
    fig, ax = plt.subplots(3, figsize=(7,7), sharex=True)
    fig.suptitle(os.path.basename(pname).split('.')[0])
    ax[0].plot(range(n_cut, n_tot), loss_D, c='k', label='loss D')
    ax[0].legend(loc='best')
    if adversary == 'MINE':
        ax[1].plot([0, n_tot], [0,0], 'k:')
        ax[1].plot(range(n_cut, n_tot), MIs, c='navy', label='True MI')
        ax[1].plot(range(n_cut, n_tot), -np.array(loss_R), c='navy', linestyle=':',  label='Estimate of MI')
        ax[1].legend(loc='best')
    else:
        ax[1].plot(range(n_cut, n_tot), loss_R, c='navy', label='loss R')
        ax[1].legend(loc='best')
    ax[2].plot(range(n_cut, n_tot), loss_DR, c='royalblue', label='loss DR')
    ax[2].legend(loc='best')
    ax[2].set_xlabel('Adversarial cycles')
    ax[2].set_xlim(0, n_tot)

    # save
    plt.savefig(pname)
    plt.close(fig)


def plot_MI(MINEs, MIs, n_adv_cycles, pname, batch=False):
    
    # plot
    fig, ax = plt.subplots(figsize=(7,7))
    fig.suptitle(os.path.basename(pname).split('.')[0])
    ax.plot([0, n_adv_cycles], [0,0], 'k:')
    ax.plot(range(len(MIs)), MIs, 'g-', label='True MI')
    ax.plot(range(len(MINEs)), MINEs, 'g:', label='Estimate of MI')
    ax.set_xlim(0, n_adv_cycles)
    ax.set_xlabel('Adversarial cycles')
    ax.set_ylim(-0.01)
    ax.legend(loc='best')
    
    # save
    plt.savefig(pname)
    if not batch:
        plt.show()
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


def plot_hmumu_performance(X, Y, Z, W, fX, benchmarks, pname, batch=False):

    # unpack benchmarks
    fprs, tprs, titles = benchmarks 

    # cut in sensitive parameter
    mw = 10 # 10 GeV mass window on each side
    ind_lo = Z < 125-mw
    ind_hi = Z > 125+mw
    ind_mi = np.logical_and(np.logical_not(ind_lo), np.logical_not(ind_hi))
    
    X_lo_all, Y_lo_all, Z_lo_all, W_lo_all, fX_lo_all = X[ind_lo], Y[ind_lo], Z[ind_lo], W[ind_lo], fX[ind_lo]
    X_hi_all, Y_hi_all, Z_hi_all, W_hi_all, fX_hi_all = X[ind_hi], Y[ind_hi], Z[ind_hi], W[ind_hi], fX[ind_hi]
    X_mi_all, Y_mi_all, Z_mi_all, W_mi_all, fX_mi_all = X[ind_mi], Y[ind_mi], Z[ind_mi], W[ind_mi], fX[ind_mi]
    
    # ROC curve
    fig, ax = plt.subplots(3, 2, figsize=(10,15))
    fig.suptitle(os.path.basename(pname).split('.')[0], fontsize=10)
    fpr_all, tpr_all, _ = roc_curve(Y, fX, sample_weight=W)
    fpr_hi, tpr_hi, _ = roc_curve(Y_hi_all, fX_hi_all, sample_weight=W_hi_all)
    fpr_mi, tpr_mi, _ = roc_curve(Y_mi_all, fX_mi_all, sample_weight=W_mi_all)
    fpr_lo, tpr_lo, _ = roc_curve(Y_lo_all, fX_lo_all, sample_weight=W_lo_all)
    ax[0,0].plot([0,1], [1,0], 'k:', label='Random guess')
    for k in fprs.keys():
        ax[0,0].plot(1-fprs[k], tprs[k], 'k:', label=titles[k])
    ax[0,0].plot(1-fpr_all, tpr_all, linestyle='-', c='red', label='All')
    ax[0,0].plot(1-fpr_hi, tpr_hi, linestyle=':', c='darkred', label='High mass')
    ax[0,0].plot(1-fpr_mi, tpr_mi, linestyle=':', c='red', label='Mid mass')
    ax[0,0].plot(1-fpr_lo, tpr_lo, linestyle=':', c='tomato', label='Low mass')
    ax[0,0].set_xlabel('Background rejection')
    ax[0,0].set_ylabel('Signal efficiency')
    ax[0,0].legend(loc='best')
    
    for yval in [0, 1]:
        
        # select only signal/background events
        i_hi = Y_hi_all == yval
        i_mi = Y_mi_all == yval
        i_lo = Y_lo_all == yval
        X_hi, Y_hi, Z_hi, W_hi, fX_hi = X_hi_all[i_hi], Y_hi_all[i_hi], Z_hi_all[i_hi], W_hi_all[i_hi], fX_hi_all[i_hi]
        X_mi, Y_mi, Z_mi, W_mi, fX_mi = X_mi_all[i_mi], Y_mi_all[i_mi], Z_mi_all[i_mi], W_mi_all[i_mi], fX_mi_all[i_mi]
        X_lo, Y_lo, Z_lo, W_lo, fX_lo = X_lo_all[i_lo], Y_lo_all[i_lo], Z_lo_all[i_lo], W_lo_all[i_lo], fX_lo_all[i_lo]
 
        y_text = 'signal' if yval == 1 else 'background'
    
        # set parameters
        nbins = 50
        mass_lo = 110
        mass_hi = 160
    
        # invariant mass plot
        ax[yval+1, 0].set_title(y_text+' only events')
        ax[yval+1, 0].hist(Z_hi, bins=nbins, weights=W_hi, range=(mass_lo,mass_hi), histtype='step', color='darkred', label='High mass')
        ax[yval+1, 0].hist(Z_mi, bins=nbins, weights=W_mi, range=(mass_lo,mass_hi), histtype='step', color='red', label='Mid mass')
        ax[yval+1, 0].hist(Z_lo, bins=nbins, weights=W_lo, range=(mass_lo,mass_hi), histtype='step', color='tomato', label='Low mass')
        ax[yval+1, 0].set_xlim(mass_lo, mass_hi)
        ax[yval+1, 0].set_xlabel('Invariant Mass [GeV]')
        ax[yval+1, 0].legend(loc='best')
    
        # classifier response distributions
        ax[yval+1, 1].set_title(y_text+' only events')
        ax[yval+1, 1].hist(fX_hi, bins=nbins, weights=W_hi, range=(0,1), histtype='step', density=True, color='darkred', label='High mass')
        ax[yval+1, 1].hist(fX_mi, bins=nbins, weights=W_mi, range=(0,1), histtype='step', density=True, color='red', label='Mid mass')
        ax[yval+1, 1].hist(fX_lo, bins=nbins, weights=W_lo, range=(0,1), histtype='step', density=True, color='tomato', label='Low mass')
        ax[yval+1, 1].set_xlabel('Classifier output')
        ax[yval+1, 1].legend(loc='best')
    
    # save
    plt.savefig(pname)
    print(pname)
    if not batch:
        plt.show()
    plt.close(fig)


def plot_var_sets(benchmarks, nets, pname, batch=False):

    # unpack inputs
    fprs, tprs, labels = benchmarks
    nfprs, ntprs, nlabels = nets
    var_sets = fprs.keys()
    
    # plot
    fig, ax = plt.subplots(figsize=(7,7))
    fig.suptitle(os.path.basename(pname).split('.')[0], fontsize=10)
    lstyles = {'low':':', 'high':'--', 'both':'-'}
    for v in var_sets:
        ax.plot(1-fprs[v], tprs[v], label=labels[v], c='k', linestyle=lstyles[v])
        ax.plot(1-nfprs[v], ntprs[v], label=nlabels[v], c=utils.blue, linestyle=lstyles[v])
    ax.legend(loc='best')
    ax.set_xlabel('Background rejection')
    ax.set_ylabel('Signal efficiency')
    
    # save
    plt.savefig(pname)
    print(pname)
    if not batch:
        plt.show()
    plt.close(fig)


def plot_spurious_signal(Z_ss, test_package, preds_GBC_ss, preds_DNN_ss, percentile, path, batch=False):
    
    ####################
    # compute spurious signal
    ####################

    # get var list
    var_sets = list(Z_ss.keys())

    # get the mass distros
    cut_GBC, cut_DNN, GBC_Zs, DNN_Zs = {}, {}, {}, {}
    for v in var_sets:

        # reshape
        preds_DNN_ss[v] = preds_DNN_ss[v].reshape(-1) # column to row

        # get the percentile values
        cut_GBC[v] = np.percentile(preds_GBC_ss[v], 100-percentile)
        cut_DNN[v] = np.percentile(preds_DNN_ss[v], 100-percentile)

        # get the mass distros
        GBC_Zs[v] = Z_ss[v][preds_GBC_ss[v] > cut_GBC[v]]
        DNN_Zs[v] = Z_ss[v][preds_DNN_ss[v] > cut_DNN[v]]

    # unpack the data and mc
    X, Y, Z, W, preds_GBC, preds_DNN = test_package 
    nsel_GBC_sig, nsel_DNN_sig = {}, {}
    nsel_GBC_data, nsel_DNN_data = {}, {} # total number of data events, extrapolated from the sidebands

    for v in var_sets:
 
        # reshape DNN
        preds_DNN[v] = preds_DNN[v].reshape(-1)

        # get the number of selected signal events
        preds_GBC_sig = preds_GBC[v][Y[v] == 1]
        preds_DNN_sig = preds_DNN[v][Y[v] == 1]

        weights_signal = W[v][Y[v] == 1]

        mask_GBC_signal = preds_GBC_sig > cut_GBC[v]
        mask_DNN_signal = preds_DNN_sig > cut_DNN[v]

        nsel_GBC_sig[v] = np.sum(weights_signal[mask_GBC_signal])
        nsel_DNN_sig[v] = np.sum(weights_signal[mask_DNN_signal])

        # get the number of selected data events (sideband only)
        preds_GBC_data = preds_GBC[v][Y[v] == 0]
        preds_DNN_data = preds_DNN[v][Y[v] == 0]

        weights_data = W[v][Y[v] == 0] # all 1

        mask_GBC_data = preds_GBC_data > cut_GBC[v]
        mask_DNN_data = preds_DNN_data > cut_DNN[v]

        nsel_GBC_data[v] = np.sum(weights_data[mask_GBC_data])
        nsel_DNN_data[v] = np.sum(weights_data[mask_DNN_data])

        # and extrapolate to the signal region using ss MC
        nsel_GBC_ss = len(GBC_Zs[v])
        nsel_DNN_ss = len(DNN_Zs[v])

        nsideband_GBC_ss = np.sum( (GBC_Zs[v] < 120) | (GBC_Zs[v] > 130) )
        nsideband_DNN_ss = np.sum( (DNN_Zs[v] < 120) | (DNN_Zs[v] > 130) )

        tf_GBC = nsel_GBC_ss / float(nsideband_GBC_ss)
        tf_DNN = nsel_DNN_ss / float(nsideband_DNN_ss)

        nsel_GBC_data[v] *= tf_GBC
        nsel_DNN_data[v] *= tf_DNN

    ####################
    # make the plot
    ####################

    # plot
    fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    fig.suptitle(os.path.basename(path).split('.')[0], fontsize=10)

    # top panel
    lstyles = {'low':':', 'high':'--', 'both':'-'}
    xlow, xhigh = 110, 160
    bins = 100
    GBC_hist, DNN_hist = {}, {}
    for v in var_sets:
        common_kwargs = {'bins':bins, 'range':(xlow,xhigh), 'histtype':'step', 'linestyle':lstyles[v]}
        GBC_hist[v], _, _ = ax[0].hist(GBC_Zs[v], color='k', label='GBC ({}) best {}%'.format(v, percentile), **common_kwargs)
        DNN_hist[v], _, _ = ax[0].hist(DNN_Zs[v], color=utils.blue, label='DNN ({}) best {}%'.format(v, percentile), **common_kwargs)
    ax[0].set_title('Background-only MC')
    ax[0].legend(loc='best')
    ax[0].set_xlim(xlow, xhigh)
    ax[0].set_ylabel('Selected events')

    # bottom panel
    Z_hist, edges = np.histogram(Z_ss[var_sets[0]], bins=bins, range=(xlow,xhigh))
    lows = edges[:-1]
    highs = edges[1:]
    centres = (lows+highs)*0.5

    ax[1].plot([xlow, xhigh], [percentile/100., percentile/100.], 'k:')
    for v in var_sets:
        common_kwargs = {'bins':bins, 'range':(xlow,xhigh), 'histtype':'step', 'linestyle':lstyles[v]}
        ax[1].hist(centres, weights=GBC_hist[v]/Z_hist, color='k', **common_kwargs)
        ax[1].hist(centres, weights=DNN_hist[v]/Z_hist, color=utils.blue, **common_kwargs)
    ax[1].set_xlim(xlow, xhigh)
    ax[1].set_ylabel('Selected/All ratio')
    ax[1].set_xlabel('Invariant Mass [GeV]')

    # save
    plt.savefig(path)
    print(path)
    if not batch:
        plt.show()
    plt.close(fig)




