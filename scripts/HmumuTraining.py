import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier

import sys
sys.path.insert(0, os.path.abspath('../fairml'))
import utils
import plotting
import models
import actions
import generate as G


def main():

    # parse the arguments
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--n-epochs',
                        type=int,
                        default=5,
                        help='Number of epochs to train on.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.005,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.001,
                        help='Epsilon controls the Adam algorithm oscillation near convergence.')
    parser.add_argument('--adversary',
                        default=None,
                        choices=[None, 'MINE', 'GaussMixNLL'],
                        help='What to use as the adversary.')
    parser.add_argument('--lam',
                        type=float,
                        default=50.,
                        help='Lambda controls the adversary cost.')
    args = parser.parse_args()

    #####################
    # Hyperparameters and settings
    #####################

    description = utils.dict_to_unix(vars(args))

    # hyperparameters
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    lam = args.lam
    deep = True

    n_samples = 1000 # batch size
    n_components = 5 # components of gaussian mixture surrogate model
    n_clf = 1 
    n_adv = 5
    name = 'model'

    #####################
    # Generate test data
    #####################

    var_sets = ['low', 'high', 'both']
    
    # get the scalers, and the convenince function
    x_scaler, z_scaler, generate = {}, {}, {}
    
    for v in var_sets:
        x_scaler[v], z_scaler[v], generate[v] = G.generate_hmumu(features=v)
    
    # generate test data (a large, one time only thing)
    n_test_samples = 100000 # TODO: have 'all' option in the generate function
    X, Y, Z, W, Z_plot = {}, {}, {}, {}, {}
    
    for v in var_sets:
        X[v], Y[v], Z[v], W[v] = generate[v](n_test_samples, balanced=False) # need Global weights for testing
        Z_plot[v] = z_scaler[v].inverse_transform(Z[v], copy=True)
    
    #####################
    # Create GBC benchmarks
    #####################

    n_train_samples = 10000 # TODO: have 'all' option in the generate function
    
    # create training samples
    X_train, Y_train, W_train = {}, {}, {}
    for v in var_sets:
        X_train[v], Y_train[v], _, W_train[v] = generate[v](n_train_samples, balanced=True)
    
    # train
    gbc400, preds400, fpr400, tpr400 = {}, {}, {}, {}
    
    for v in var_sets:
    
        # train benchmarks
        gbc400[v] = GradientBoostingClassifier(n_estimators=400)
        gbc400[v].fit(X_train[v], Y_train[v], sample_weight=W_train[v])
        preds400[v] = gbc400[v].predict_proba(X[v])[:, 1]
    
        # roc curve
        fpr400[v], tpr400[v], _ = roc_curve(Y[v], preds400[v], sample_weight=W[v])
    
    # save the benchmarks for the performance comparison
    fprs = {v:fpr400[v] for v in var_sets}
    tprs = {v:tpr400[v] for v in var_sets}
    labels = {v:'GBC400 ({})'.format(v) for v in var_sets}
    benchmarks = fprs, tprs, labels

    #####################
    # prepare the graphs
    #####################

    # input placeholders
    x_in, y_in, z_in, w_in, inputs = {}, {}, {}, {}, {}
    for v in var_sets:
        x_in[v] = tf.placeholder(tf.float32, shape=(None, X[v].shape[1]), name='X'+v)
        y_in[v] = tf.placeholder(tf.float32, shape=(None, ), name='Y'+v)
        z_in[v] = tf.placeholder(tf.float32, shape=(None, ), name='Z'+v)
        w_in[v] = tf.placeholder(tf.float32, shape=(None, ), name='W'+v)
        inputs[v] = [x_in[v], y_in[v], z_in[v], w_in[v]]
    
    # create the classifier graph, loss, and optimisation
    clf_output, vars_D, loss_D, opt_D = {}, {}, {}, {}
    for v in var_sets:
        clf_output[v], vars_D[v] = models.classifier(x_in[v], name+'_clf'+v, deep=deep)
        loss_D[v] = models.classifier_loss(clf_output[v], y_in[v], w_in[v])
        opt_D[v] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_D[v], var_list=vars_D[v])
    
    #####################
    # start the training
    #####################

    # initialise the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    # train the classifiers
    for e in range(n_epochs):
        
        # report on progress
        if e%10 == 0:
            print('{}/{}'.format(e, n_epochs))
        
        # training step and roc curve compuation
        npreds, nfprs, ntprs, nlabels = {}, {}, {}, {}
        for v in var_sets:
            actions.train(sess, opt_D[v], loss_D[v], inputs[v], generate[v], n_samples, 1, None)
            npreds[v] = utils.sigmoid(sess.run(clf_output[v], feed_dict={x_in[v]:X[v]}))
            nfprs[v], ntprs[v], _ = roc_curve(Y[v], npreds[v], sample_weight=W[v])
            nlabels[v] = 'NN {}'.format(v)
        
        nets = nfprs, ntprs, nlabels
        
        # make the variable comparison plot
        pname = 'VarsComparison'
        dirn = 'media/plots/{}'.format(pname)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        path = '{d}/{n}_{c:03}.png'.format(d=dirn, n=description, c=e)
        plotting.plot_var_sets(benchmarks, nets, path, batch=True)

        # make the classifier performance plot
        pname = 'MassCheck'
        dirn = 'media/plots/{}'.format(pname)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        path = '{d}/{n}_{c:03}.png'.format(d=dirn, n=description, c=e)
        v = 'both'
        plotting.plot_hmumu_performance(X[v], Y[v], Z_plot[v], W[v], npreds[v].reshape(-1), benchmarks, path, batch=True)

    #####################
    # make the gif out of the plots
    #####################

    if not os.path.exists('media/gifs'):
        os.makedirs('media/gifs')

    for pname in ['VarsComparison', 'MassCheck']:
        dirn = 'media/plots/{}'.format(pname)
        in_pngs = ' '.join(['{d}/{n}_{c:03}.png'.format(d=dirn, n=description, c=c) for c in range(n_epochs)])
        out_gif = 'media/gifs/{p}_{n}_{c}.gif'.format(p=pname, n=description, c=n_epochs)
        os.system('convert -colors 32 -loop 0 -delay 10 {i} {o}'.format(i=in_pngs, o=out_gif))
        print(out_gif)


if __name__ == '__main__':
    main()
