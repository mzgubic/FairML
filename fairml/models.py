import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def MINE(x_in, y_in, name, H=10, deep=False):

    # reshape the tensor to correct shape [if (100, ) reshape to (100, 1)]
    n_samples = tf.shape(y_in)[0]
    y_in = tf.reshape(y_in, shape=(n_samples, 1))
    
    # use scoped names 
    with tf.variable_scope(name):

        # shuffle the y in (independent prabability density)
        y_shuffle = tf.random_shuffle(y_in)
        x_conc = tf.concat([x_in, x_in], axis=0)
        y_conc = tf.concat([y_in, y_shuffle], axis=0)

        # compute the forward pass
        layerx = layers.linear(x_conc, H)
        layery = layers.linear(y_conc, H)
        layer2 = tf.nn.relu(layerx + layery)
        if deep:
            layer2 = layers.relu(layer2, H)
            layer2 = layers.relu(layer2, H)
        output = layers.linear(layer2, 1)

        # split in the T_xy and T_x_y predictions
        N_batch = tf.shape(x_in)[0]
        T_xy = output[:N_batch]
        T_x_y = output[N_batch:]

    # get the variables
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

    return T_xy, T_x_y, tf_vars


def MINE_loss(T_xy, T_x_y):
    
    # compute the loss
    loss = - (tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis=0)))
    return loss


def classifier(x_in, name):
    
    with tf.variable_scope(name):
        
        # define the output of the network
        dense1 = layers.relu(x_in, 20)
        dense2 = layers.relu(dense1, 20)
        output = layers.linear(dense2, 1)

    these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    
    return output, these_vars


def classifier_loss(clf_output, y_in, w_in=None):
    
    # define the loss 
    n_samples = tf.shape(y_in)[0]
    y_shaped = tf.reshape(y_in, shape=(n_samples, 1))
    if w_in == None:
        loss_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_shaped, logits=clf_output))
    else:
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_shaped, logits=clf_output)
        loss_D = tf.reduce_mean(w_in * xentropy)
    
    return loss_D


def adversary_gaussmix(clf_output, n_components, name):
    
    with tf.variable_scope(name):
        
        # define the output of a network (depends on number of components)
        dense1 = layers.relu(clf_output, 20)
        dense2 = layers.relu(dense1, 20)
        output_noact = layers.linear(dense2, 3*n_components)
        
        # make sure sigmas are positive and pis are normalised 
        mu = output_noact[:, :n_components]
        sigma = tf.exp(output_noact[:, n_components:2*n_components])
        pi = tf.nn.softmax(output_noact[:, 2*n_components:])
        
        # and merge them together again
        output = tf.concat([mu, sigma, pi], axis=1)
    
    these_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    
    return output, these_vars


def adversary_gaussmix_loss(z_in, adv_output, n_components):
    
    # build the pdf (max likelihood principle)
    mu = adv_output[:, :n_components]
    sigma = adv_output[:, n_components:2*n_components]
    pi = adv_output[:, 2*n_components:]
    
    pdf = 0
    for c in range(n_components):
        pdf += pi[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
                tf.math.exp(-(z_in - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))
            
    # make the loss
    nll = - tf.math.log(pdf)
    loss_R = tf.reduce_mean(nll)
    
    return loss_R
