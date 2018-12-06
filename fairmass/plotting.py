import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def plot_classifier_performance(x_in, y_in, clf_output, generate, sess):
        
    def sigmoid(x):
        return 1 / (1 + np.e**(-x))

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
    ax[1,1].hist(pred1, bins=bins, normed=True, color='darkred', histtype='step', label='Z=1')
    ax[1,1].hist(pred0, bins=bins, normed=True, color='red', histtype='step', label='Z=0')
    ax[1,1].hist(pred_1, bins=bins, normed=True, color='tomato', histtype='step', label='Z=-1')
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
    fig.show()
    plt.savefig('classifier.pdf')

