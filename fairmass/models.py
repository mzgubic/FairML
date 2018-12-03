import tensorflow as tf
import tensorflow.contrib.layers as layers

def MINE(x_in, y_in, name, H=10, deep=False):

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

        # compute the loss
        neg_loss = - (tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis=0)))

        # optimisation step
        opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(neg_loss)

    # get the variables
    tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

    return neg_loss, opt, tf_vars

