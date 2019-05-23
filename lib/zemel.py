import numpy as np
import tensorflow as tf
from metrics import accuracy_np, accuracy_tf, discrimination_np, discrimination_tf, k_nearest_neighbors_sp, consistency_np
"""
The purpose of this class is to reproduce in Tensorflow the model proposed by:
Zemel, Rich, et al. "Learning fair representations." International Conference on Machine Learning. 2013.
https://www.cs.toronto.edu/~toni/Papers/icml-final.pdf
"""

class ZemelFairRepresentations:
    
    def __init__(
        self,
        x_dim,
        num_clusters,
        sigmas = None,
        var_scope = 'ZFR',
        dtype=tf.float64,
        optimizer=tf.train.AdamOptimizer,
    ):
        self.x_dim = x_dim
        self.num_clusters = num_clusters
        if sigmas is None:
            self.sigmas = 1.
        else:
            self.sigmas = sigmas
        self.var_scope = var_scope
        self.dtype = dtype
        self.optimizer = optimizer
        self.build_network()
    
    def build_network(self):
        dtype = self.dtype
        K = self.num_clusters
        with tf.variable_scope(self.var_scope):
            self.X_tf = X_tf = tf.placeholder(shape=[None, self.x_dim], dtype=dtype)
            self.Y_tf = Y_tf = tf.placeholder(shape=[None, 1], dtype=dtype)
            self.A_tf = A_tf = tf.placeholder(shape=[None, 2], dtype=dtype)
            self.V_tf = V_tf = tf.get_variable(name='V', shape=[self.x_dim, K], dtype=dtype)
            self.logit_W_tf = logit_W_tf = tf.get_variable(name='W', shape=[K, 1], dtype=dtype)
            self.W_tf = W_tf = 1. / (1. + tf.exp(-logit_W_tf))
            
            X_V_square_distance = tf.reduce_sum(tf.square(
                tf.tile(tf.reshape(X_tf, [-1, 1, self.x_dim]), [1, K, 1])
                - tf.transpose(V_tf)
            ) / self.sigmas, axis=2, keepdims=False)
            self.M_tf = M_tf = tf.nn.softmax(-X_V_square_distance, axis=1)
            self.X_bar = X_bar = tf.matmul(M_tf, tf.transpose(V_tf))
            self.Y_hat_tf = Y_hat_tf = tf.matmul(M_tf, W_tf)
            
            cluster_memberships = (tf.matmul(tf.transpose(M_tf), A_tf)
                                   / tf.tile(tf.reduce_sum(A_tf, axis=0, keepdims=True), [K, 1]))

            self.L_Z = L_Z = tf.reduce_sum(tf.abs(cluster_memberships[:, 0] - cluster_memberships[:, 1]))
            self.L_X = L_X = tf.reduce_mean(tf.reduce_sum(tf.square(X_tf - X_bar) / self.sigmas, axis = 1))
            self.L_Y = L_Y = - tf.reduce_mean(Y_tf * tf.log(Y_hat_tf) + (1. - Y_tf) * tf.log(1. - Y_hat_tf))
            
            self.alpha_Z = alpha_Z = tf.placeholder_with_default(tf.constant(1., dtype=dtype), shape=[])
            self.alpha_X = alpha_X = tf.placeholder_with_default(tf.constant(1., dtype=dtype), shape=[])
            self.alpha_Y = alpha_Y = tf.placeholder_with_default(tf.constant(1., dtype=dtype), shape=[])
            self.loss = loss = (alpha_Z * L_Z + alpha_X * L_X + alpha_Y * L_Y)

            self.result_tensors = [Y_hat_tf, [loss, L_Z, L_X, L_Y]]
            self.learning_rate_ph = tf.placeholder_with_default(tf.constant(1e-2, dtype=dtype), shape=[])
            self.optimizer_instance = self.optimizer(learning_rate=self.learning_rate_ph)
            self.train_op = self.optimizer_instance.minimize(loss=loss, var_list=[V_tf, logit_W_tf])
            self.initialized = False
            self.dYdX = tf.gradients(self.Y_hat_tf, self.X_tf)
    
    def initialize(self, sess=None):
        if sess is None:
            sess = self.sess
        self.sess.run(tf.variables_initializer([self.V_tf, self.logit_W_tf]))
        self.sess.run(tf.variables_initializer(self.optimizer_instance.variables()))
        self.V_np, self.logit_W_np = self.sess.run([self.V_tf, self.logit_W_tf])
        self.initialized = True
    
    def fit(self, 
            X_np, 
            Y_np, 
            A_np,
            alpha_Z = 1., 
            alpha_X = 1., 
            alpha_Y = 1.,
            n_steps = 1000,
            learning_rate = 1e-2,
            sess = None,
           ):
        
        if sess is None:
            self.sess = sess = tf.Session()
        else:
            self.sess = sess
        
        if self.initialized == False:
            self.initialize()
        
        for step in range(n_steps):
            sess.run(self.train_op, 
                     feed_dict={self.X_tf: X_np, self.Y_tf: Y_np, self.A_tf: A_np,
                                self.alpha_Z: alpha_Z, self.alpha_X: alpha_X, self.alpha_Y: alpha_Y,
                                self.learning_rate_ph: learning_rate})
        self.V_np, self.logit_W_np, self.W_np = sess.run([self.V_tf, self.logit_W_tf, self.W_tf])
        return_vals = sess.run(self.result_tensors, 
                               feed_dict={self.X_tf: X_np, self.Y_tf: Y_np, self.A_tf: A_np,
                                          self.alpha_Z: alpha_Z, self.alpha_X: alpha_X, self.alpha_Y: alpha_Y})
        return return_vals
    
    def predict(self, X_np):
        with tf.variable_scope(self.var_scope):
            with tf.Session() as sess:
                self.V_tf.load(self.V_np)
                self.logit_W_tf.load(self.logit_W_np)
                Y_hat_np = sess.run(self.Y_hat_tf, 
                                    feed_dict={self.X_tf: X_np})
        return Y_hat_np
    
    def predict_and_losses(
        self, 
        X_np, 
        Y_np, 
        A_np, 
        alpha_Z = 1., 
        alpha_X = 1., 
        alpha_Y = 1.,
    ):
        with tf.variable_scope(self.var_scope):
            with tf.Session() as sess:
                self.V_tf.load(self.V_np)
                self.logit_W_tf.load(self.logit_W_np)
                return_vals = sess.run(self.result_tensors, 
                                       feed_dict={self.X_tf: X_np, self.Y_tf: Y_np, self.A_tf: A_np,
                                                  self.alpha_Z: alpha_Z, self.alpha_X: alpha_X, self.alpha_Y: alpha_Y})
        return return_vals
        
    def zemel_metrics(self, X_np, Y_np, A_np, Yhat_np, nearest_neighbors_sp):
        return [
            accuracy_np(Y_np, Yhat_np),
            discrimination_np(A_np, Yhat_np),
            consistency_np(Yhat_np, nearest_neighbors_sp)
        ]
    
    def set_weights(self, V_np, W_np):
        self.V_np = V_np
        self.W_np = W_np
        self.logit_W_np = np.log(W_np / (1. - W_np))
        
        self.V_tf.load(self.V_np, self.sess)
        self.logit_W_tf.load(self.logit_W_np, self.sess)

    def compute_sample_gradients(self, X_np):
        return self.sess.run(self.dYdX, feed_dict = {self.X_tf: X_np})[0]

    def compute_sample_lipschitz_bound(self, X_np):
        grads = self.compute_sample_gradients(X_np)
        return np.sqrt(np.square(grads).sum(1)).max()
