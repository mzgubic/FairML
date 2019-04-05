import uuid
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import models
import plot


class TFEnvironment:
    
    def __init__(self, generate, name):

        # save variables
        self.generate = generate
        self.name = name
        self._xyz = ['X', 'Y', 'Z']

        # record losses
        self.history = {'L_clf':[], 'L_adv':[], 'L_comb':[],
                        'KS1':[], 'KS_1':[], 'KSp1':[], 'KSp_1':[], 'KS':[],
                        'auroc1':[], 'auroc0':[], 'auroc_1':[], 'auroc-mean':[], 'auroc-std':[]}

        # start the session
        print('Welcome to {} TensorFlow environment'.format(self.name))
        self._start_session()

    def _start_session(self):

        # start the session
        config=tf.ConfigProto(intra_op_parallelism_threads = 32,
                                 inter_op_parallelism_threads = 32,
                                 allow_soft_placement = True,
                                 device_count = {'CPU': 2})

        self.sess = tf.Session(config=config)

    def build_graph(self, clf_settings, adv_settings):
        
        print('--- Building computational graphs')

        # build the inputs
        batch = self.generate(10)
        self._in = {}
        for xyz in self._xyz:
            tftype = tf.int32 if xyz == 'Y' else tf.float32
            self._in[xyz] = tf.placeholder(tftype, shape=(None, batch[xyz].shape[1]), name='{}_in'.format(xyz))

        # build the classifier graph
        self.clf = models.Classifier(name=self.name+'_clf', **clf_settings)
        self.clf.build_forward(self._in['X'])

        # build the adversary graph
        self.adv = models.Adversary.create(self.name, adv_settings)

    def build_loss(self):

        print('--- Building computational graphs for losses')

        # classifier loss
        self.clf.build_loss(self._in['Y'])

        # adversary loss
        self.adv.build_loss(self.clf.output, self._in['Z'])

    def build_opt_deprecated(self, lam=1.0, opt_type='AdamOptimizer', learning_rate=0.05, projection=False):
        
        print('--- Building computational graphs for optimisations')
        
        # optimizer type
        opt = getattr(tf.train, opt_type)
        self.optimizer = opt(learning_rate=learning_rate)
        self.lam = lam
        
        # compute the optimisation steps
        self.opt_clf = self.optimizer.minimize(self.clf.loss, var_list=self.clf.tf_vars)
        self.opt_adv = self.optimizer.minimize(self.adv.loss, var_list=self.adv.tf_vars)
        comb_loss = self.clf.loss - lam * self.adv.loss
        self.opt_comb = self.optimizer.minimize(comb_loss, var_list=self.clf.tf_vars)

    def build_opt(self, lam=1.0, opt_type='AdamOptimizer', learning_rate=0.05, projection=False):

        print('--- Building computational graphs for optimisations')

        # optimizer type
        opt = getattr(tf.train, opt_type)
        self.optimizer = opt(learning_rate=learning_rate)
        self.lam = lam

        # compute the gradients (d(Loss_clf)/d(weights_clf))
        grad_Lc_clf = self.optimizer.compute_gradients(self.clf.loss, var_list=self.clf.tf_vars)
        grad_La_clf = self.optimizer.compute_gradients(self.adv.loss, var_list=self.clf.tf_vars)
        grad_La_adv = self.optimizer.compute_gradients(self.adv.loss, var_list=self.adv.tf_vars)

        grad_comb = [] # the adversarial part
        for i in range(len(grad_Lc_clf)):

            # combine the gradients
            g = grad_Lc_clf[i][0]         # direction to improve classifier
            h = - lam * grad_La_clf[i][0] # direction to make classifier fairer w.r.t to adversary
            c = g + h                     # combined direction

            # add the projection term if necessary
            if projection:
                normalize = lambda x: x / (tf.norm(x) + np.finfo(np.float32).tiny)
                h_norm = normalize(h)
                proj = tf.reduce_sum(h_norm * g) * h_norm
                c -= proj

            # and finally build the combined gradients list
            var = grad_Lc_clf[i][1]
            grad_comb.append((c, var))

        # compute the optimisation steps
        self.opt_clf = self.optimizer.apply_gradients(grad_Lc_clf)
        self.opt_adv = self.optimizer.apply_gradients(grad_La_adv)
        self.opt_comb = self.optimizer.apply_gradients(grad_comb)

    def initialise_variables(self):

        print('--- Initialising TensorFlow variables')

        self.sess.run(tf.global_variables_initializer())

    def _get_feed_dict(self, batch):

        return {self._in[xyz]:batch[xyz] for xyz in self._xyz}

    def pretrain_step_clf(self, batch_size, write=False):

        # pre-training the classifier (no adversary)
        batch = self.generate(batch_size)
        feed_dict = self._get_feed_dict(batch)
        self.sess.run(self.opt_clf, feed_dict=feed_dict)

        # update history
        if write:
            self._write_history(batch_size)

    def train_step_adv(self, batch_size, write=False):

        # train the adversary
        batch = self.generate(batch_size)
        feed_dict = self._get_feed_dict(batch)
        self.sess.run(self.opt_adv, feed_dict=feed_dict)
        
        # update history
        if write:
            self._write_history(batch_size)
    
    def train_step_clf(self, batch_size, write=True):
        
        # train the classifier (adversarially)
        batch = self.generate(batch_size)
        feed_dict = self._get_feed_dict(batch)
        self.sess.run(self.opt_comb, feed_dict=feed_dict)
        
        # update history
        if write:
            self._write_history(batch_size)
    
    def _write_history(self, batch_size):
        
        try:
            # get current losses
            batch = self.generate(batch_size)
            loss_clf, loss_adv, loss_comb = self._losses(batch)
            
            # get current metrics
            ks1, ks_1 = self._ks_metric(batch_size)
            
            # get performance metrics
            auroc1, auroc0, auroc_1 = self._roc_auc(batch_size)
            
            # append
            self.history['L_clf'].append(loss_clf)
            self.history['L_adv'].append(loss_adv)
            self.history['L_comb'].append(loss_comb)
            self.history['KS1'].append(ks1[0])
            self.history['KSp1'].append(ks1[1])
            self.history['KS_1'].append(ks_1[0])
            self.history['KSp_1'].append(ks_1[1])
            self.history['KS'].append(np.mean([ks1[0], ks_1[0]]))
            self.history['auroc1'].append(auroc1)
            self.history['auroc0'].append(auroc0)
            self.history['auroc_1'].append(auroc_1)
            self.history['auroc-mean'].append(np.mean([auroc1, auroc0, auroc_1]))
            self.history['auroc-std'].append(np.std([auroc1, auroc0, auroc_1]))
        except ValueError:
            print('ValueError caught, training probably failed')
    
    def _losses(self, batch):
        
        feed_dict = self._get_feed_dict(batch)
        loss_clf, loss_adv = self.sess.run([self.clf.loss, self.adv.loss], feed_dict=feed_dict)
        return loss_clf, loss_adv, loss_clf - self.lam * loss_adv
    
    def _ks_metric(self, batch_size):
        
        # get predictions
        preds = {}
        for z in [1, 0, -1]:
            batch = self.generate(batch_size, z=z)
            preds[z] = self.predict(batch)
        
        # compute the metrics
        ks1 = scipy.stats.ks_2samp(preds[1].ravel(), preds[0].ravel())
        ks_1 = scipy.stats.ks_2samp(preds[-1].ravel(), preds[0].ravel())
        
        return ks1, ks_1
    
    def _roc_auc(self, batch_size):
        
        # get predictions
        batch = {}
        preds = {}
        for z in [1, 0, -1]:
            batch[z] = self.generate(batch_size, z=z)
            preds[z] = self.predict(batch[z])
            
        # compute AUROC
        auroc1 = roc_auc_score(batch[1]['Y'].ravel(), preds[1].ravel())
        auroc0 = roc_auc_score(batch[0]['Y'].ravel(), preds[0].ravel())
        auroc_1 = roc_auc_score(batch[-1]['Y'].ravel(), preds[-1].ravel())
        
        return auroc1, auroc0, auroc_1

    def _pdf_pars(self, batch_size):

        # get predictions
        batch = self.generate(batch_size)
        feed_dict = self._get_feed_dict(batch)

        return self.sess.run(self.adv.nll_pars, feed_dict=feed_dict)
        
    def predict(self, batch):
        
        feed_dict = self._get_feed_dict(batch)
        preds = self.sess.run(self.clf.output, feed_dict=feed_dict)
        
        return preds

    def show_performance(self, batch_size):
        
        print('--- Plot classifier performance and fairness')
        
        try:
            # get predictions
            batch = {}
            preds = {}
            for z in [None, 1, 0, -1]:
                batch[z] = self.generate(batch_size, z=z)
                preds[z] = self.predict(batch[z])
            
            # prepare the figure
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle('Performance for {} (lambda={})'.format(self.adv.__class__.__name__, self.lam))
            
            # plot the variates
            plot.variates_main(ax[0,0], batch[None])
            
            # plot the ROC curves
            plot.roc_curves(ax[1,0], batch[1], batch[0], batch[-1], preds[1], preds[0], preds[-1])
            
            # plot the classifier output
            plot.clf_outputs(ax[1,1], preds[1], preds[0], preds[-1])
            
            # plot the decision boundary
            plot.decision_boundary(ax[0,1], batch[None], preds[None])
            
            # show
            fig.show()
        except ValueError:
            print('ValueError caught, probably not converged')
        
    def show_history(self):
        
        remove_first = 0
        
        # prepare the plot
        fig, ax = plt.subplots(8, figsize=(7,16), sharex=True)
        fig.suptitle('Losses for {} (lambda={})'.format(self.adv.__class__.__name__, self.lam))
        
        # classifier loss
        plot.history(ax[0], self.history['L_clf'], '-', 'royalblue', 'Classifier Loss', remove_first)
        
        # adversary loss
        plot.history(ax[1], self.history['L_adv'], '-', 'crimson', 'Adversary loss', remove_first)
        
        # combined loss
        plot.history(ax[2], self.history['L_comb'], '-', 'k', 'Combined loss', remove_first)
        
        # accuracy
        plot.history(ax[3], self.history['auroc1'], '-', 'darkblue', 'AUROC (z=1)', remove_first)
        plot.history(ax[3], self.history['auroc0'], '-', 'royalblue', 'AUROC (z=0)', remove_first)
        plot.history(ax[3], self.history['auroc_1'], '-', 'lightskyblue', 'AUROC (z=-1)', remove_first)

        # summary accuracy (mean)
        plot.history(ax[4], self.history['auroc-mean'], '-', 'royalblue', 'mean(AUROC), Z={1,0,-1}', remove_first)

        # summary accuracy (std)
        plot.history(ax[5], self.history['auroc-std'], ':', 'royalblue', 'std(AUROC), Z={1,0,-1}', remove_first)

        # KS metric
        plot.history(ax[6], self.history['KS1'], '-', 'darkred', 'KS (z=1, z=0)')
        plot.history(ax[6], self.history['KSp1'], ':', 'darkred', 'p-value (z=1, z=0)')
        plot.history(ax[6], self.history['KS_1'], '-', 'tomato', 'KS (z=-1, z=0)')
        plot.history(ax[6], self.history['KSp_1'], ':', 'tomato', 'p-value (z=-1, z=0)')

        # summary KS metric
        plot.history(ax[7], self.history['KS'], '-', 'red', 'KS metric', remove_first)
        
        # cosmetics
        for i in range(len(ax)):
            ax[i].legend(loc='best')
        ax[4].set_xlabel('Training steps')


def bootcamp(N, generate_toys, clf_settings, adv_settings, opt_settings, trn_settings):

    # train a collection of environments
    envs = []
    for i in range(N):

        # make the environment
        tfe = TFEnvironment(generate_toys, 'env_{}'.format(uuid.uuid4()))
        tfe.build_graph(clf_settings, adv_settings)
        tfe.build_loss()
        tfe.build_opt(**opt_settings)
        tfe.initialise_variables()

        # pretrain
        batch_size = trn_settings['batch_size']
        n_pretrain = trn_settings['n_pretrain']
        for _ in range(n_pretrain):
            tfe.pretrain_step_clf(batch_size, write=False)
        for _ in range(n_pretrain):
            tfe.train_step_adv(batch_size, write=False)
    
        # train
        n_train = trn_settings['n_train']
        for _ in range(n_train):

            tfe.train_step_clf(batch_size)
            for __ in range(5):
                tfe.train_step_adv(batch_size)
            
        # append to envs
        envs.append(tfe)
    
    # return the collection
    return envs

