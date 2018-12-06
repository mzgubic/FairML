import numpy as np
import matplotlib.pyplot as plt


def train(sess, opt, loss, inputs, generate, n_samples, n_epochs, text):
    
    # unpack
    x_in, y_in, z_in = inputs
    
    # train it
    losses = []
    
    for epoch in range(n_epochs):
        X, Y, Z = generate(n_samples)
        feed_dict = {x_in:X, y_in:Y, z_in:Z}
        _, l = sess.run([opt, loss], feed_dict=feed_dict)
        losses.append(l)
    
    # plot the losses
    if not text==None:
        fig, ax = plt.subplots()
        ax.plot(range(len(losses)), losses, c='k', label=text)
        ax.set_xlabel('epochs')
        ax.legend(loc='best')
        ax.set_title(text)
        plt.show()
    
    # return the value of the loss
    try:
        return np.mean(losses[-5:])
    except IndexError:
        return losses[-1]


def train_adversarially(sess, losses, opts, inputs, generate, n_samples, n_adv_cycles=50, n_clf=1, n_adv=5, text=None):
    
    # unbox inputs, losses and opts
    x_in, y_in, z_in = inputs
    loss_D, loss_R, loss_DR = losses
    _, opt_R, opt_DR = opts
    
    # keep track of losses
    losses = {'D':[], 'R':[], 'DR':[]}

    for e in range(n_adv_cycles):
    
        if e%(n_adv_cycles//10) == 0:
            print('{e}/{t}'.format(e=e, t=n_adv_cycles))
        # optimisation steps
        losses['DR'].append(train(sess, opt_DR, loss_DR, inputs, generate, n_samples, n_clf, None))
        losses['R'].append(train(sess, opt_R, loss_R, inputs, generate, n_samples, n_adv, None))
    
        # monitor only
        X, Y, Z = generate(n_samples)
        losses['D'].append(sess.run(loss_D, feed_dict={x_in:X, y_in:Y, z_in:Z}))
    
    if not text==None:
        fig, ax = plt.subplots(3, sharex=True)
        ax[0].plot(range(len(losses['DR'])), losses['DR'], c='k', label='Loss_DR')
        ax[0].legend(loc='best')
        ax[1].plot(range(len(losses['R'])), losses['R'], c='r', label='Loss_R')
        ax[1].legend(loc='best')
        ax[2].plot(range(len(losses['D'])), losses['D'], c='b', label='Loss_D')
        ax[2].legend(loc='best')
        ax[2].set_xlabel('Adversarial cycles')
        ax[0].set_title('Losses for {} Adversary'.format(text))
        plt.show()
