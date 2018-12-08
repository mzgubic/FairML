import os
import argparse
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from sklearn.feature_selection import mutual_info_regression

import sys
sys.path.insert(0, '/Users/zgubic/Projects/FairMass/fairmass')
import plotting
import generate
import models
import actions
import utils

def main():

    # parse the arguments
    parser = argparse.ArgumentParser(description='ArgParser')
    parser.add_argument('--adversary',
                        type=str,
                        default='MINE',
                        choices=['MINE', 'GaussMixNLL'],
                        help='What to use as the adversary.')
    parser.add_argument('--n-adv-cycles',
                        type=int,
                        default=5,
                        help='Number of adversarial cycles to run on.')
    parser.add_argument('--lam',
                        type=float,
                        default=50.,
                        help='Lambda controls the adversary cost.')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.01,
                        help='Learning rate for the optimiser.')
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.001,
                        help='Epsilon controls the Adam algorithm oscillation near convergence.')
    args = parser.parse_args()

    description = utils.dict_to_unix(vars(args))

    #####################
    # generate test data (a large, one time only thing)
    #####################
    n_test_samples = 100000

    X, Y, Z = generate.generate_toys(n_test_samples)
    X1, Y1, Z1 = generate.generate_toys(n_test_samples, z=1)
    X0, Y0, Z0 = generate.generate_toys(n_test_samples, z=0)
    X_1, Y_1, Z_1 = generate.generate_toys(n_test_samples, z=-1)

    test_data = {}
    test_data['all Z'] = X, Y, Z
    test_data['Z=1'] = X1, Y1, Z1
    test_data['Z=0'] = X0, Y0, Z0
    test_data['Z=-1'] = X_1, Y_1, Z_1
    
    #####################
    # start the session
    #####################

    sess = tf.InteractiveSession()
    ctr = 0
    
    #  hyperparameters
    n_samples = 5000 # training stats in each step
    n_epochs = 30 # pretraining steps
    n_components = 5 # components of gaussian mixture surrogate model
    n_adv_cycles = args.n_adv_cycles 
    n_clf = 1 
    n_adv = 5
    lam = args.lam
    ctr+=1
    name = 'model'
        
    #####################
    # prepare the classifier, adversary, losses, and optimisation steps
    #####################

    # input placeholders
    x_in = tf.placeholder(tf.float32, shape=(None, 2), name='X12')
    y_in = tf.placeholder(tf.float32, shape=(None, ), name='Y')
    z_in = tf.placeholder(tf.float32, shape=(None, ), name='Z')
    inputs = [x_in, y_in, z_in]
    
    # create the classifier graph, loss, and optimisation
    clf_output, vars_D = models.classifier(x_in, name+'_clf')
    loss_D = models.classifier_loss(clf_output, y_in)
    opt_D = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon).minimize(loss_D, var_list=vars_D)
    
    # create the adversary graph, loss, and optimisation
    if args.adversary == 'MINE':
        T_xy, T_x_y, vars_R = models.MINE(clf_output, z_in, name+'_adv')
        loss_R = models.MINE_loss(T_xy, T_x_y)
        opt_R = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon).minimize(loss_R, var_list=vars_R)

    elif args.adversary == 'GaussMixNLL':
        adv_output, vars_R = models.adversary_gaussmix(clf_output, n_components, name+'_adv')
        loss_R = models.adversary_gaussmix_loss(z_in, adv_output, n_components)
        opt_R = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon).minimize(loss_R, var_list=vars_R)
    
    # create the combined loss function (for the classifier training in the adversarial part)
    loss_DR = loss_D - lam*loss_R
    opt_DR = tf.train.AdamOptimizer(learning_rate=args.learning_rate, epsilon=args.epsilon).minimize(loss_DR, var_list=vars_D)
    
    # initialise the variables
    sess.run(tf.global_variables_initializer())
    
    # pretrain the classifier and the adversary
    actions.train(sess, opt_D, loss_D, inputs, generate.generate_toys, n_samples, n_epochs, None)
    actions.train(sess, opt_R, loss_R, inputs, generate.generate_toys, n_samples, n_epochs, None)
    
    #####################
    # Train adversarially
    #####################

    losses = [loss_D, loss_R, loss_DR]
    opts = [None, opt_R, opt_DR]
    feed_dict = {x_in:X, y_in:Y, z_in:Z}
    
    plot_names = set()
    loss_vals = {'l_D':[], 'l_R':[], 'l_DR':[]}
    MIs = []
    for cycle in range(n_adv_cycles):
        
        # training step
        print('{}/{}'.format(1+cycle, n_adv_cycles))
        actions.train_adversarially(sess, losses, opts, inputs, generate.generate_toys, n_samples, 1, n_clf, n_adv, None)
        
        # evaluate the graphs
        l_D, l_R, l_DR, fX = sess.run([loss_D, loss_R, loss_DR, clf_output], feed_dict=feed_dict)
        MIs.append(mutual_info_regression(fX, Z)[0])
        loss_vals['l_D'].append(l_D)
        loss_vals['l_R'].append(l_R)
        loss_vals['l_DR'].append(l_DR)
        fX = fX.reshape(-1)

        #####################
        # Classifier performance plot
        #####################
        
        # name of the plot
        plot_name = 'ClfPerf__'+description
        plot_names.add(plot_name)
        dirn = 'media/plots/{}'.format(plot_name)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        path = '{d}/{n}_{c:03}.png'.format(d=dirn, n=plot_name, c=cycle)
        
        # make the plot
        pred = utils.sigmoid(sess.run(clf_output, feed_dict={x_in:X, y_in:Y}))
        pred1 = utils.sigmoid(sess.run(clf_output, feed_dict={x_in:X1, y_in:Y1}))
        pred0 = utils.sigmoid(sess.run(clf_output, feed_dict={x_in:X0, y_in:Y0}))
        pred_1 = utils.sigmoid(sess.run(clf_output, feed_dict={x_in:X_1, y_in:Y_1}))
        test_data['preds'] = pred, pred1, pred0, pred_1
        plotting.plot_classifier_performance(test_data, path, batch=True)
        
        #####################
        # 2D points plot
        #####################
        
        # name of the plot
        plot_name = 'fX_vs_Z__'+description
        plot_names.add(plot_name)
        dirn = 'media/plots/{}'.format(plot_name)
        if not os.path.exists(dirn):
            os.makedirs(dirn)
        path = '{d}/{n}_{c:03}.png'.format(d=dirn, n=plot_name, c=cycle)
        
        # make the plot
        plotting.plot_2D(pred, Y, Z, path, batch=True)

    # plot the losses
    dirn = 'media/plots/losses'
    if not os.path.exists(dirn):
        os.makedirs(dirn)
    path = '{d}/Losses_{p}.png'.format(d=dirn, p=description)
    plotting.plot_losses(loss_vals['l_D'], loss_vals['l_R'], loss_vals['l_DR'], MIs, args.adversary, path)
    print(path)
    
    # animate the plots to a gif
    for pname in plot_names:
        if not os.path.exists('media/gifs'):
            os.makedirs('media/gifs')
        dirn = 'media/plots/{}'.format(pname)
        in_pngs = ' '.join(['{d}/{p}_{c:03}.png'.format(d=dirn, p=pname, c=c) for c in range(n_adv_cycles)])
        out_gif = 'media/gifs/{p}_{c}.gif'.format(p=pname, c=n_adv_cycles)
        os.system('convert -colors 16 -loop 0 -delay 10 {i} {o}'.format(i=in_pngs, o=out_gif))
        print(out_gif)

if __name__ == '__main__':
    main()
