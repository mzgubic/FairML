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
sys.path.insert(0, os.path.abspath(os.path.join(os.getenv('PROJECT_DIR'), 'fairml')))
import utils
import plotting
import models
import actions
import generate as G


def train(args):

    #####################
    # Hyperparameters and settings
    #####################

    description = utils.dict_to_unix(vars(args))
    print(description)

    # hyperparameters
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    lam = args.lam
    deep = True

    n_pretrain_epochs = 30
    n_samples = 1000 # batch size
    n_components = 5 # components of gaussian mixture surrogate model
    n_clf = 1 
    n_adv = 5
    name = 'model'

    var_sets = ['low', 'high', 'both']
    var_sets = ['both']

    #####################
    # Generate test data (performance and spurious signal)
    #####################
    print('--- Generate test data')

    # get the scalers, and the convenince function
    x_scaler, z_scaler, generate, generate_ss = {}, {}, {}, {}
    
    for v in var_sets:
        x_scaler[v], z_scaler[v], generate[v] = G.generate_hmumu(features=v)
        generate_ss[v] = G.generate_ss(features=v, x_scaler=x_scaler[v], z_scaler=z_scaler[v])
    
    # generate test data (a large, one time only thing)
    n_test_samples = 100000 # TODO: have 'all' option in the generate function
    n_spur_samples = 100000
    X, Y, Z, W, Z_plot = {}, {}, {}, {}, {}
    X_ss, Y_ss, Z_ss, W_ss, Z_ss_plot = {}, {}, {}, {}, {}
    
    for v in var_sets:
        X[v], Y[v], Z[v], W[v] = generate[v](n_test_samples, balanced=False, train=False) # need Global weights for testing
        Z_plot[v] = z_scaler[v].inverse_transform(Z[v], copy=True)
        X_ss[v], Y_ss[v], Z_ss[v], W_ss[v] = generate_ss[v](n_spur_samples, train=True) # training set is 80%
        Z_ss_plot[v] = z_scaler[v].inverse_transform(Z_ss[v], copy=True)
    
    #####################
    # Create GBC benchmarks
    #####################

    n_train_samples = 10000 # TODO: have 'all' option in the generate function
    
    # create training samples
    X_train, Y_train, W_train = {}, {}, {}
    for v in var_sets:
        X_train[v], Y_train[v], _, W_train[v] = generate[v](n_train_samples, balanced=True)
    
    # train
    gbc400, preds400, preds400_ss, fpr400, tpr400 = {}, {}, {}, {}, {}
    
    for v in var_sets:
    
        # train benchmarks
        gbc400[v] = GradientBoostingClassifier(n_estimators=400)
        gbc400[v].fit(X_train[v], Y_train[v], sample_weight=W_train[v])
        preds400[v] = gbc400[v].predict_proba(X[v])[:, 1]
        preds400_ss[v] = gbc400[v].predict_proba(X_ss[v])[:, 1]
    
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
    print('--- Prepare graphs')

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

    # create the adversary graphs, loss, and optimisations
    adv_output, vars_R, loss_R, opt_R, loss_DR, opt_DR = {}, {}, {}, {}, {}, {}
    for v in var_sets:

        # choose your adversary
        if args.adversary == 'GaussMixNLL':
            adv_output[v], vars_R[v] = models.adversary_gaussmix(clf_output[v], n_components, name+'_adv'+v)
            loss_R[v] = models.adversary_gaussmix_loss(z_in[v], adv_output[v], n_components)

        elif args.adversary == 'MINE':
            pass

        elif args.adversary == 'None':
            break # the for loop

        # optimisations and losses
        opt_R[v] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_R[v], var_list=vars_R[v])
        loss_DR[v] = loss_D[v] - lam*loss_R[v]
        opt_DR[v] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_DR[v], var_list=vars_D[v])

    #####################
    # pretrain the classifier and adversary
    #####################
    print('--- Pretrain classifiers and adversaries.')

    # initialise the variables
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    if not args.adversary == 'None':
        
        for v in var_sets:
            actions.train(sess, opt_D[v], loss_D[v], inputs[v], generate[v], n_samples, n_pretrain_epochs, None)
            actions.train(sess, opt_R[v], loss_R[v], inputs[v], generate[v], n_samples, n_pretrain_epochs, None)

    #####################
    # start the training
    #####################  
    print('--- Training')

    # train the classifiers
    for e in range(n_epochs):
        
        # training step and roc curve computation
        npreds, nfprs, ntprs, nlabels, npreds_ss = {}, {}, {}, {}, {}
        for v in var_sets:

            if args.adversary == 'None':
                actions.train(sess, opt_D[v], loss_D[v], inputs[v], generate[v], n_samples, 1, None)
            else:
                actions.train(sess, opt_DR[v], loss_DR[v], inputs[v], generate[v], n_samples, n_clf, None)
                actions.train(sess, opt_R[v], loss_R[v], inputs[v], generate[v], n_samples, n_adv, None)

            # run on test
            npreds[v] = utils.sigmoid(sess.run(clf_output[v], feed_dict={x_in[v]:X[v]}))
            nfprs[v], ntprs[v], _ = roc_curve(Y[v], npreds[v], sample_weight=W[v])
            nlabels[v] = 'DNN ({})'.format(v)

            # run on ss
            npreds_ss[v] = utils.sigmoid(sess.run(clf_output[v], feed_dict={x_in[v]:X_ss[v]}))
        
        # pack ROC curves for the neural net
        nets = nfprs, ntprs, nlabels
        
        # every ten training steps report on progress and make a plot
        if e%10 == 0:

            # report and set variables to both low and high level (only plot those)
            print('{}/{}'.format(e, n_epochs))
            v = 'both'

            def get_path(pname, c):
                dirn = 'media/plots/{p}/{d}'.format(p=pname, d=description)
                if not os.path.exists(dirn):
                    os.makedirs(dirn)
                path = '{d}/{p}_{n}_{c:03}.pdf'.format(d=dirn, p=pname, n=description, c=e)
                return path
        
            # make the classifier performance plot
            #path = get_path('MassCheck', e)
            #plotting.plot_hmumu_performance(X[v], Y[v], Z_plot[v], W[v], npreds[v].reshape(-1), benchmarks, path, batch=True)

            # make the variables comparison plot
            path = get_path('VarsComparison', e)
            plotting.plot_var_sets(benchmarks, nets, path, batch=True)

            # make the spurious signal test plot
            percentiles = [50]
            for p in percentiles:
                path = get_path('SpuriousSignal{}'.format(p), e)
                test_package = X, Y, Z, W, preds400, npreds
                plotting.plot_spurious_signal(Z_ss_plot, test_package, preds400_ss, npreds_ss, p, path, batch=True)

    #####################
    # make the gif out of the plots
    #####################

    if False:
        if not os.path.exists('media/gifs'):
            os.makedirs('media/gifs')
    
        for pname in ['MassCheck']:
            dirn = 'media/plots/{p}/{d}'.format(p=pname, d=description)
            in_pngs = ' '.join(['{d}/{n}_{c:03}.png'.format(d=dirn, n=description, c=c) for c in range(n_epochs) if c%10==0])
            out_gif = 'media/gifs/{p}_{n}.gif'.format(p=pname, n=description)
            os.system('convert -colors 32 -loop 0 -delay 10 {i} {o}'.format(i=in_pngs, o=out_gif))
            print(out_gif)


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
                        default='None',
                        choices=['None', 'MINE', 'GaussMixNLL'],
                        help='What to use as the adversary.')
    parser.add_argument('--lam',
                        type=float,
                        default=50.,
                        help='Lambda controls the adversary cost.')
    parser.add_argument('--batch',
                        action='store_true',
                        default=False,
                        help='Send to batch or run interactively.')
    args = parser.parse_args()

    # run
    if args.batch == False:
        train(args)

    # or send a job
    else:

        # base command
        command = 'python3 '+os.path.join(utils.PROJ, 'scripts/HmumuTraining.py')

        # add the arguments
        for arg in args.__dict__:

            if arg == 'batch':
                continue

            key = arg.replace('_', '-')
            command += ' --{k} {v}'.format(k=key, v=args.__dict__[arg])

        commands = ['cd scripts', command]
        utils.submit_commands(commands, queue='veryshort', job_name='training')


if __name__ == '__main__':
    main()
