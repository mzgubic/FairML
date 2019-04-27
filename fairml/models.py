import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


class Model:

    def __init__(self, name, depth, width):

        self.depth = depth
        self.width = width
        self.name = name


class Classifier(Model):

    def __init__(self, name, depth=2, width=20):

        super().__init__(name, depth, width)
        self.n_classes = 2

    @classmethod
    def create(cls, name, clf_settings):
        
        clf_type = clf_settings['type']
        
        classes = {'Linear':LinearClassifier,
                   'DNN':DNNClassifier}
        
        # check if implemented
        if clf_type not in classes:
            raise ValueError('Unknown Classifier type {}.'.format(clf_type))
        
        # return the right one
        classifier = classes[clf_type]
        kwargs = clf_settings.copy()
        kwargs.pop('type')
        return classifier(name='{}_{}_clf'.format(name, clf_type), **kwargs)


class DNNClassifier(Classifier):

    def __init__(self, name, depth, width):

        super().__init__(name, depth, width)

    def build_forward(self, x_in):

        with tf.variable_scope(self.name):

            # input layer
            layer = x_in

            # hidden layers
            for _ in range(self.depth):
                layer = layers.relu(layer, self.width)

            # logits and output
            self.logits = layers.linear(layer, self.n_classes)
            self.output = tf.reshape(layers.softmax(self.logits)[:, 1], shape=(-1, 1))

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    def build_loss(self, labels):

        print('--- Building classifier loss')

        # one hot encode the labels
        one_hot = tf.one_hot(tf.reshape(labels, shape=[-1]), depth=self.n_classes)

        # and build the loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot, logits=self.logits))
        

class LinearClassifier(DNNClassifier):

    def __init__(self, name, **kwargs):

        super().__init__(name, 0, 0)


class Adversary(Model):
    
    def __init__(self, name, depth=2, width=20):
        
        super().__init__(name, depth, width)
        self.loss = None
        self.tf_vars = None
    
    @classmethod
    def create(cls, name, adv_settings):
        
        adv_type = adv_settings['type']
        
        classes = {'Dummy':DummyAdversary,
                   'GMM':GMMAdversary,
                   'MINE':MINEAdversary,
                   'PtEst':PtEstAdversary}
        
        # check if implemented
        if adv_type not in classes:
            raise ValueError('Unknown Adversary type {}.'.format(adv_type))
        
        # return the right one
        adversary = classes[adv_type]
        kwargs = adv_settings.copy()
        kwargs.pop('type')
        return adversary(name='{}_{}_adv'.format(name, adv_type), **kwargs)


class PtEstAdversary(Adversary):

    def __init__(self, name, depth=2, width=20, **kwargs):

        super().__init__(name, depth, width, **kwargs)
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    def build_loss(self, fX, Z):

        # forward pass
        with tf.variable_scope(self.name):

            # input layer
            layer = fX

            # hidden layers
            for _ in range(self.depth):
                layer = layers.relu(layer, self.width)

            # output layer
            self.output = layers.linear(layer, 1)

        # variables
        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

        # create the loss
        self.loss = tf.reduce_mean((self.output - Z)**2)


class DummyAdversary(Adversary):
    
    def __init__(self, name, **kwargs):
        
        super().__init__(name, **kwargs)
        
    def build_loss(self, fX, Z):
        
        with tf.variable_scope(self.name):
            dummy_var = tf.Variable(0.1, name='dummy')
            self.loss = dummy_var**2 # i.e. goes to zero
            self.loss += 0 * tf.reduce_mean(fX) # and connects to the classifier weights

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


class GMMAdversary(Adversary):
    
    def __init__(self, name, depth=2, width=20, n_components=5, **kwargs):
        
        super().__init__(name, depth, width, **kwargs)
        
        self.nll_pars = None
        self.n_components = n_components
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
    
    def build_loss(self, fX, Z):

        # nll network
        self._make_nll(fX)
        
        # loss
        self._make_loss(Z)
    
    def _make_nll(self, fX):

        print('--- Building GMM nll model')
        
        n_components = self.n_components
        
        with tf.variable_scope(self.name):

            # define the input layer
            layer = fX

            # define the output of a network (depends on number of components)
            for _ in range(self.depth):
                layer = layers.relu(layer, self.width)

            # output layer: (mu, sigma, amplitude) for each component
            output = layers.linear(layer, 3*n_components)

            # make sure sigmas are positive and pis are normalised
            mu = output[:, :n_components]
            sigma = tf.exp(output[:, n_components:2*n_components])
            pi = tf.nn.softmax(output[:, 2*n_components:])

            # interpret the output layers as nll parameters
            self.nll_pars = tf.concat([mu, sigma, pi], axis=1)

        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    
    def _make_loss(self, Z):
        
        print('--- Building GMM loss')
        
        # for convenience
        n_components = self.n_components

        # build the pdf (max likelihood principle)
        mu = self.nll_pars[:, :n_components]
        sigma = self.nll_pars[:, n_components:2*n_components]
        pi = self.nll_pars[:, 2*n_components:]

        likelihood = 0
        for c in range(n_components):

            # normalisation
            norm_vec = tf.reshape(pi[:, c] * (1. / np.sqrt(2. * np.pi)) / sigma[:, c], shape=(-1, 1))

            # exponential
            mu_vec = tf.reshape(mu[:, c], shape=(-1, 1))
            sigma_vec = tf.reshape(sigma[:, c], shape=(-1, 1))
            exp = tf.math.exp(-(Z - mu_vec) ** 2 / (2. * sigma_vec ** 2))

            # add to likelihood
            likelihood += norm_vec * exp

        # make the loss
        nll = - tf.math.log(likelihood)
        self.loss = tf.reduce_mean(nll)
        

class MINEAdversary(Adversary):
    
    def __init__(self, name, depth=2, width=10, **kwargs):
        
        super().__init__(name, depth, width, **kwargs)
        
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])
            
    def build_loss(self, fX, Z):

        print('--- Building MINE loss')
        
        # store the input placeholders
        fX = tf.reshape(fX, shape=(-1, 1))
        Z = tf.reshape(Z, shape=(-1, 1))

        # aliases
        x_in = fX
        y_in = Z

        # use scope to keep track of vars
        with tf.variable_scope(self.name):
            
            # shuffle one of them
            y_shuffle = tf.random_shuffle(y_in)
            x_conc = tf.concat([x_in, x_in], axis=0)
            y_conc = tf.concat([y_in, y_shuffle], axis=0)

            # compute the forward pass
            layer_x = layers.linear(x_conc, self.width)
            layer_y = layers.linear(y_conc, self.width)
            layer = tf.nn.relu(layer_x + layer_y)

            for _ in range(self.depth):
                layer = layers.relu(layer, self.width)

            output = layers.linear(layer, 1)

            # split in T_xy and T_x_y
            N_batch = tf.shape(x_in)[0]
            T_xy = output[:N_batch]
            T_x_y = output[N_batch:]

            # compute the loss
            #self.loss = - (tf.reduce_mean(T_xy, axis=0) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y), axis=0)))
            self.loss = - (tf.reduce_mean(T_xy) - tf.math.log(tf.reduce_mean(tf.math.exp(T_x_y))))

        # save variables
        self.tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)


