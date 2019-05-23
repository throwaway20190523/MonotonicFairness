import numpy as np
import tensorflow as tf
import warnings

def monotone_ffnn(input_data, 
                  num_layers=5, 
                  width=[10, 10, 10, 10, 10],
                  output_dim=1, 
                  activations=[tf.tanh], 
                  activate_last_layer=True, 
                  var_scope="FFNN",
                  reuse=None,
                  enforce_monotonicty=True,
                  monotonicity=None, # Should be list like [1, 0, -1] for [monotone inc, non-monotone, monoton dec.]
                  positive_func= lambda w: 1. + tf.nn.elu(w - 1.),
                  dtype=None,
                 ):
    """Create or reuse a sub-variable_scope to implement a MONOTONE feedforward neural network, with arguments:
      input_data: a tensorflow array (constant, variable, or placeholder) of shape [batch_size, input_dim]
      num_layers: how many layers deep the network should be
      width: can be:
          - a single integer, in which case all layers will be width * len(activations) long
          - a len(activations)-length list of integers, in which case all layers will have sum(width) nodes where width[a] 
              nodes use activations[a]
          - a num_layers-length list of len(activations)-length lists of integers, in case each layer l will have 
              sum(width[l]) nodes where width[l][a] nodes use activation[a]
          NOTE: if activate_last_layer is True, then the implied number of nodes for the final layer must match the 
              specified output_dim!
      output_dim: the desired dimension of each row of the output (provide a single integer)
      activations: a list of tensorflow functions that will transform the data at each layer.
      activate_last_layer: a boolean to denote whether to provide transformed or untransformed output of the final 
          layer.  Note that numerical stability can sometimes be improved by post-processing untransformed data.
      scope: character string to use as the name of the sub-variable_scope
      reuse: whether to re-use an existing scope or create a new one.  If left blank, will only create if necessary 
          and re-use otherse.
      Returns: a tf node of the output layer
    """
    # Reading some implicit figures
    if isinstance(input_data, tf.Tensor):
        batch_size, input_dim = input_data._shape_as_list()
    elif isinstance(input_data, tf.Variable):
        batch_size, input_dim = input_data.shape
    else:
        assert False, 'Input should be a tf.Tensor or tf.Variable'
    
    # If variable re-use hasn't been specified, figure out if the scope is in use and should be re-used
    if reuse == None:
        local_scope = tf.get_variable_scope().name + '/' + var_scope
        scope_in_use = max([obj.name[:len(local_scope)] == local_scope for obj in tf.global_variables()] + [False])
        reuse = scope_in_use
        if scope_in_use == True:
            warnings.warn('Automatically Re-using variables for ' + local_scope + ' scope')
    
    # Process the width and activation inputs into useable numbers
    if isinstance(width, list):
        if isinstance(width[0], list):
            layer_widths = width
        else:
            layer_widths = [width] * num_layers
    else:
        layer_widths = [[width] * len(activations)] * num_layers
    
    # Check for coherency of final layer if needed
    if activate_last_layer and sum(layer_widths[-1]) != output_dim:
        print layer_widths
        raise BaseException(
            'activate_last_layer == True but implied final layer width doesn\'t match output_dim \n (implied depth: {}, explicit depth: {}'.format(
                sum(layer_widths[num_layers - 1]),
                output_dim,
            )
        )
    
    # Set-up/retrieve the appropriate nodes within scope
    with tf.variable_scope(var_scope, reuse=reuse) as vs:
        pre_Ws = [tf.get_variable("W_" + str(l),
                              shape=[input_dim if l == 0 else sum(layer_widths[l - 1]),
                                     output_dim if l == num_layers - 1 else sum(layer_widths[l])],
                              dtype=input_data.dtype)
              for l in range(num_layers)]
        
        ############################################
        ############################################
        # This is the critical part for monotonicity
        if enforce_monotonicty:
            Ws = [tf.abs(W) for W in pre_Ws]
            if monotonicity is not None:
                W = pre_Ws[0]
                Ws[0] = tf.convert_to_tensor([
                               positive_func(pre_Ws[0][d, :]) if monotonicity[d] ==  1
                    else -1. * positive_func(pre_Ws[0][d, :]) if monotonicity[d] == -1
                    else                     pre_Ws[0][d, :]  #if monotonicity[d] ==  0
                    for d in range(input_dim)
                ])
        else:
            Ws = [W for W in pre_Ws]
        ############################################
        ############################################
        
        Bs = [tf.get_variable("B_" + str(l), shape=[output_dim if l == num_layers - 1 else sum(layer_widths[l])],
                              dtype=input_data.dtype) for l in range(num_layers)]
        Hs = [None] * num_layers
        HLs = [None] * num_layers
        for l in range(num_layers):
            # print(l, (input_data if l == 0 else Hs[l - 1]).shape, Ws[l].shape, Bs[l].shape)
            h_in = input_data if l == 0 else Hs[l - 1]
            HLs[l] = tf.matmul(h_in, Ws[l]) + Bs[l]
            if l < num_layers - 1 or activate_last_layer == True:
                Hs[l] = tf.concat(
                    [activations[a](HLs[l][:, sum(layer_widths[l][0:a]):sum(layer_widths[l][0:a + 1])]) for a in
                     range(len(activations))], 1)
            else:
                Hs[l] = HLs[l]
    
    variables = pre_Ws + Bs
    return Hs[l], variables


def ffnn(
    input_data, 
    num_layers=5, 
    width=[10, 10, 10, 10, 10], 
    output_dim=1, 
    activations=[tf.tanh], 
    activate_last_layer=True, 
    var_scope="FFNN",
    reuse=None,
):
    """Create or reuse a sub-variable_scope to implement a feedforward neural network, with arguments:
      input_data: a tensorflow array (constant, variable, or placeholder) of shape [batch_size, input_dim]
      num_layers: how many layers deep the network should be
      width: can be:
          - a single integer, in which case all layers will be width * len(activations) long
          - a len(activations)-length list of integers, in which case all layers will have sum(width) nodes where width[a] 
              nodes use activations[a]
          - a num_layers-length list of len(activations)-length lists of integers, in case each layer l will have 
              sum(width[l]) nodes where width[l][a] nodes use activation[a]
          NOTE: if activate_last_layer is True, then the implied number of nodes for the final layer must match the 
              specified output_dim!
      output_dim: the desired dimension of each row of the output (provide a single integer)
      activations: a list of tensorflow functions that will transform the data at each layer.
      activate_last_layer: a boolean to denote whether to provide transformed or untransformed output of the final 
          layer.  Note that numerical stability can sometimes be improved by post-processing untransformed data.
      scope: character string to use as the name of the sub-variable_scope
      reuse: whether to re-use an existing scope or create a new one.  If left blank, will only create if necessary 
          and re-use otherse.
      Returns: a tf node of the output layer
    """
    # Reading some implicit figures
    batch_size, input_dim = input_data._shape_as_list()
    
    # If variable re-use hasn't been specified, figure out if the scope is in use and should be re-used
    if reuse == None:
        local_scope = tf.get_variable_scope().name + '/' + var_scope
        scope_in_use = max([obj.name[:len(local_scope)] == local_scope for obj in tf.global_variables()] + [False])
        reuse = scope_in_use
        if scope_in_use == True:
            warnings.warn('Automatically Re-using variables for ' + local_scope + ' scope')
    
    # Process the width and activation inputs into useable numbers
    if isinstance(width, list):
        if isinstance(width[0], list):
            layer_widths = width
        else:
            layer_widths = [width] * num_layers
    else:
        layer_widths = [[width] * len(activations)] * num_layers
    
    # Check for coherency of final layer if needed
    if activate_last_layer and sum(layer_widths[-1]) != output_dim:
        print layer_widths
        raise BaseException(
            'activate_last_layer == True but implied final layer width doesn\'t match output_dim \n (implied depth: {}, explicit depth: {}'.format(
                sum(layer_widths[num_layers - 1]),
                output_dim,
            )
        )
    
    # Set-up/retrieve the appropriate nodes within scope
    with tf.variable_scope(var_scope, reuse=reuse) as vs:
        Ws = [tf.get_variable("W_" + str(l),
                              shape=[input_dim if l == 0 else sum(layer_widths[l - 1]),
                                     output_dim if l == num_layers - 1 else sum(layer_widths[l])],
                              dtype=input_data.dtype)
              for l in range(num_layers)]
        Bs = [tf.get_variable("B_" + str(l), shape=[output_dim if l == num_layers - 1 else sum(layer_widths[l])],
                              dtype=input_data.dtype) for l in range(num_layers)]
        Hs = [None] * num_layers
        HLs = [None] * num_layers
        for l in range(num_layers):
            HLs[l] = tf.add(tf.matmul(input_data if l == 0 else Hs[l - 1], Ws[l]), Bs[l])
            if l < num_layers - 1 or activate_last_layer == True:
                Hs[l] = tf.concat(
                    [activations[a](HLs[l][:, sum(layer_widths[l][0:a]):sum(layer_widths[l][0:a + 1])]) for a in
                     range(len(activations))], 1)
            else:
                Hs[l] = HLs[l]
    
    variables = tf.contrib.framework.get_variables(vs)
    return Hs[l], variables


class FeedforwardNeuralNetwork:
    def __init__(
        self,
        x_dim,
        dtype=tf.float64,
        num_layers=5, 
        width=[10, 10, 10, 10, 10], 
        output_dim=1, 
        activations=[tf.tanh], 
        activate_last_layer=True, 
        var_scope='ffnn',
    ):
        self.x_dim = x_dim
        self.nn_params = {
            'num_layers': num_layers,
            'width': width,
            'output_dim': output_dim,
            'activations': activations,
            'activate_last_layer': activate_last_layer,
            'var_scope': var_scope,
        }
        self.dtype = dtype
        
        _, self.vars = ffnn(
            input_data = tf.placeholder([1, x_dim], dtype=self.dtype),
            reuse=False,
            **self.nn_params
        )
    
    def get_vars(self):
        return self.vars
    
    def apply(self, X):
        output, _ = ffnn(
            input_data=X,
            reuse=True,
            **self.nn_params
        )
        return output
        

def unitize_gradients(loss,
                      var_list,
                      dtype=None,
                      learning_rate_ph = None,
                      tf_train_optimizer = None,
                      min_scalar = None,
                     ):
    """Set up an optimizer which takes max-length steps"""
    if learning_rate_ph is None:
        learning_rate_ph = tf.placeholder_with_default(input=1e-2 * tf.ones([], dtype=loss.dtype), shape=[])
    
    if tf_train_optimizer is None:
        optimizer = tf_train_optimizer(learning_rate=learning_rate_ph)
    else:
        optimizer = tf_train_optimizer
        
    if min_scalar is None:
        min_scalar = 1.
    
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    scalar = tf.sqrt(tf.reduce_sum(tf.convert_to_tensor([tf.reduce_sum(tf.square(grad)) for grad in gvs])))
    scalar = tf.maximum(scalar, min_scalar)
    scaled_gvs = [(grad / scalar, var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(scaled_gvs)
    return train_op, learning_rate_ph, scalar, optimizer 


class Monotone_Feedforward_Neural_Network:
    def __init__(
        self,
        x_dim,
        dtype=tf.float64,
        num_layers=5, 
        width=[10, 10, 10, 10, 10], 
        output_dim=1, 
        activations=[tf.tanh], 
        activate_last_layer=True, 
        var_scope='mffnn',
        monotonicity=None, # Should be list like [1, 0, -1] for [monotone inc, non-monotone, monoton dec.]
        positive_func= lambda w: 1. + tf.nn.elu(w - 1.),
    ):
        self.x_dim = x_dim
        self.nn_params = {
            'num_layers': num_layers,
            'width': width,
            'output_dim': output_dim,
            'activations': activations,
            'activate_last_layer': activate_last_layer,
            'var_scope': var_scope,
            'monotonicity': monotonicity,
            'positive_func': positive_func,
        }
        self.dtype = dtype
        self.build_network()
    
    def build_network(self):
        self.X_tf = tf.placeholder(shape=[None, self.x_dim], dtype=self.dtype)
        self.Y_tf, self.vars = monotone_ffnn(
            input_data = self.X_tf,
            reuse=False,
            **self.nn_params
        )
        
    def save_vars(self, session=None):
        if session is None:
            sess = self.sess
        self.vars_np = session.run(self.vars)
        return self.vars_np
    
    def load_vars(self, session, vals=None):
        if vals is None:
            vals = self.vars_np
        for var, val in zip(self.vars, vals):
            var.load(val, session)
    
    def get_vars(self):
        return self.vars
    
    def apply(self, X):
        output, _ = ffnn(
            input_data=X,
            reuse=True,
            **self.nn_params
        )
        return output
    
    def apply_np(self, X, session, ):
        return session.run(
            self.Y_tf,
            feed_dict = {self.X_tf: X}
        )


class Fair_MNN:
    def __init__(
        self,
        x_dim,
        a_dim = 2,
        var_scope = 'fair_mnn',
        nn_params = {
            'dtype': tf.float64,
            'num_layers': 5, 
            'width': [[10], [10], [10], [10], [1]], 
            'output_dim': 1, 
            'activations': [tf.tanh], 
            'activate_last_layer': False, 
            'monotonicity': None, # None = all positive
            'positive_func': lambda w: 1. + tf.nn.elu(w - 1.),
            'var_scope': 'mffnn',
        },
        fairness_loss_type = ['DP', 'EOdds', 'EOutcome', 'EOpportunity'][0],
        y_loss_type = ['CE', 'EC'][0],
        use_minibatch = False,
        batch_size = 32,
        optimizer = tf.train.AdamOptimizer,
        X_minibatch_tf = None,
        Y_minibatch_tf = None,
        A_minibatch_tf = None,
    ):
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.var_scope = var_scope
        self.nn_params = nn_params
        self.fairness_loss_type = fairness_loss_type
        self.y_loss_type = y_loss_type
        self.optimizer = optimizer
        self.X_minibatch_tf = X_minibatch_tf
        self.Y_minibatch_tf = Y_minibatch_tf
        self.A_minibatch_tf = A_minibatch_tf
    
        self.build_network()
        
    def build_network(self):
        dtype = self.nn_params['dtype']
        with tf.variable_scope(self.var_scope):
            
            if (self.X_minibatch_tf is not None and 
                self.Y_minibatch_tf is not None and 
                self.A_minibatch_tf is not None):
                self.X_tf = X_tf = tf.placeholder_with_default(
                    input=self.X_minibatch_tf, shape=[None, self.x_dim], name='X_tf')
                self.Y_tf = Y_tf = tf.placeholder_with_default(
                    input=self.Y_minibatch_tf, shape=[None, 1],          name='Y_tf')
                self.A_tf = A_tf = tf.placeholder_with_default(
                    input=self.A_minibatch_tf, shape=[None, self.a_dim], name='A_tf')
            else:
                self.X_tf = X_tf = tf.placeholder(shape=[None, self.x_dim], dtype = dtype, name='X_tf')
                self.Y_tf = Y_tf = tf.placeholder(shape=[None, 1], dtype=dtype, name='Y_tf')
                self.A_tf = A_tf = tf.placeholder(shape=[None, self.a_dim], dtype=dtype, name='A_tf')
            self.N_0_ph = N_0_ph = tf.placeholder(shape=[], dtype=dtype, name='N_0')
            self.A_0_ph = A_0_ph = tf.placeholder(shape=[self.a_dim], dtype=dtype, name='A_0')
            self.lambda_A_ph = lambda_A_ph = tf.placeholder_with_default(
                input=tf.ones([], dtype=dtype), shape=[], name='lA')
            self.lambda_N_ph = lambda_N_ph = tf.placeholder_with_default(
                input=tf.ones([], dtype=dtype), shape=[], name='lN')
            self.lambda_Y_ph = lambda_Y_ph = tf.placeholder_with_default(
                input=tf.ones([], dtype=dtype), shape=[], name='lY')
            self.learning_rate_ph = tf.placeholder_with_default(
                input=tf.ones([], dtype=dtype), shape=[], name='lr')

            self.pre_logit_p_tf, self.vars_tf = monotone_ffnn(X_tf, reuse=False, **self.nn_params)
            logit_p_tf = self.logit_p_tf = tf.placeholder_with_default(
                input=self.pre_logit_p_tf, shape=[None, 1], name='logit_p')
            self.p_tf = p_tf = 1. / (1. + tf.exp(-logit_p_tf))

            self.e_A_tf = e_A_tf = tf.reduce_sum(p_tf * A_tf, axis=0) / tf.reduce_sum(A_tf, axis=0)
            self.e_N_tf = e_N_tf = tf.reduce_sum(p_tf) / tf.reduce_sum(tf.ones_like(p_tf))
            self.e_Y_tf = e_Y_tf = tf.reduce_sum(p_tf * Y_tf) / tf.reduce_sum(p_tf)

            N_sq_loss = tf.square(tf.reduce_mean(p_tf) - N_0_ph)

            Y_ce = -tf.reduce_mean((logit_p_tf * Y_tf) - tf.log(tf.exp(logit_p_tf) + 1.))
            Y_ec = tf.reduce_sum(Y_tf * p_tf + (1. - Y_tf) * (1. - p_tf))

            mean_p_by_group = tf.matmul(tf.transpose(p_tf), A_tf) / tf.reduce_sum(A_tf, axis=0)
            A_loss_ZemelDiscrimination = (
                tf.reduce_max(mean_p_by_group) - tf.reduce_min(mean_p_by_group)
            )
            A_loss_EOutcome = tf.reduce_sum(tf.square(e_A_tf / tf.reduce_sum(p_tf) - A_0_ph))
            A_loss_DP = tf.reduce_sum(tf.square(e_A_tf / tf.reduce_sum(A_tf, axis=0) - A_0_ph))

            ###########################################################################
            A_loss_EOpportunity = tf.reduce_mean(tf.square(
                tf.reduce_sum(Y_tf * p_tf, axis=0) / (tf.reduce_sum(Y_tf, axis=0) + 1e-8)
                - tf.reduce_sum(Y_tf * p_tf) / tf.reduce_sum(Y_tf)
            ))
            A_loss_EOdds = A_loss_EOpportunity + tf.reduce_mean(tf.square(
                tf.reduce_sum((1. - Y_tf) * p_tf, axis=0) / (tf.reduce_sum((1. - Y_tf), axis=0) + 1e-8)
                - tf.reduce_sum((1. - Y_tf) * p_tf) / tf.reduce_sum((1. - Y_tf))
            ))
            ###########################################################################

            self.N_loss = N_loss = N_sq_loss
            self.Y_loss = Y_loss = Y_ce
            if self.fairness_loss_type == 'DP':
                self.A_loss = A_loss = A_loss_DP
            elif self.fairness_loss_type == 'EOdds':
                self.A_loss = A_loss = A_loss_EOdds
            elif self.fairness_loss_type == 'EOutcome':
                self.A_loss = A_loss = A_loss_EOutcome
            elif self.fairness_loss_type == 'EOpportunity':
                self.A_loss = A_loss = A_loss_EOpportunity
            elif self.fairness_loss_type == 'ZemelDiscrimination':
                self.A_loss = A_loss = A_loss_ZemelDiscrimination
            else:
                assert False, 'fairness_loss_type not recognized'

            self.loss = loss = ( 0.
                + lambda_A_ph * A_loss
                + lambda_N_ph * N_loss
                + lambda_Y_ph * Y_loss
            )
            self.explicit_losses = {
                'N_sq_loss': N_sq_loss,
                'Y_ce': Y_ce,
                'Y_ec': Y_ec,
                'A_loss_DP': A_loss_DP,
                'A_loss_EOutcome': A_loss_EOutcome,
                'A_loss_EOpportunity': A_loss_EOpportunity,
                'A_loss_EOdds': A_loss_EOdds,
                'N_loss': N_loss,
                'A_loss': A_loss,
                'Y_loss': Y_loss,
                'loss': loss,
                'e_A': e_A_tf,
                'e_N': e_N_tf,
                'e_Y': e_Y_tf,
            }
            self.optimizer_instance = self.optimizer(learning_rate=self.learning_rate_ph)
            self.train_op = self.optimizer_instance.minimize(loss=loss, var_list=self.vars_tf)
            self.initialized = False
            self.dYdX = tf.gradients(ys = self.p_tf, xs = self.X_tf)
    
    def initialize(self, sess=None):
        if sess is None:
            sess = self.sess
        sess.run(tf.variables_initializer(self.vars_tf))
        sess.run(tf.variables_initializer(self.optimizer_instance.variables()))
        self.initialized = True
    
    def fit(
        self,
        X_np, 
        Y_np, 
        A_np,
        A_0,
        N_0,
        lambda_A = 1., 
        lambda_N = 1., 
        lambda_Y = 1.,
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
        
        feed_dict = {
            self.A_0_ph: A_0,
            self.N_0_ph: N_0,
            self.lambda_A_ph: lambda_A,
            self.lambda_N_ph: lambda_N,
            self.lambda_Y_ph: lambda_Y,
            self.learning_rate_ph: learning_rate,
        }
        
        if (X_np is not None and 
            Y_np is not None and 
            A_np is not None):
            feed_dict.update({
                self.X_tf: X_np,
                self.Y_tf: Y_np,
                self.A_tf: A_np,
            })
        for step in range(n_steps):
            sess.run(self.train_op, feed_dict = feed_dict)
        
        self.save_vars()
        losses_np = sess.run(self.explicit_losses, feed_dict = feed_dict)
        return losses_np
    
    def predict(self, X_np):
        return self.sess.run(self.p_tf, feed_dict={self.X_tf: X_np})
    
    def predict_and_losses(
        self,
        X_np, 
        Y_np, 
        A_np,
        A_0_np,
        N_0_np,
        lambda_A = 1., 
        lambda_N = 1., 
        lambda_Y = 1.,
    ):
        return self.sess.run(
            [self.logit_p_tf, self.explicit_losses],
            feed_dict = {
                self.X_tf: X_np,
                self.Y_tf: Y_np,
                self.A_tf: A_np,
                self.lambda_A_ph: lambda_A,
                self.lambda_N_ph: lambda_N,
                self.lambda_Y_ph: lambda_Y,
                self.A_0_ph: A_0_np,
                self.N_0_ph: N_0_np,
            }
        )
    
    def losses(
        self,
        X_np, 
        Y_np,
        A_np,
        logit_p_np,
        A_0_np,
        N_0_np,
        lambda_A = 1., 
        lambda_N = 1., 
        lambda_Y = 1.,
    ):
        return self.sess.run(
            self.explicit_losses,
            feed_dict = {
                self.X_tf: X_np,
                self.Y_tf: Y_np,
                self.A_tf: A_np,
                self.logit_p_tf: logit_p_np,
                self.lambda_A_ph: lambda_A,
                self.lambda_N_ph: lambda_N,
                self.lambda_Y_ph: lambda_Y,
                self.A_0_ph: A_0_np,
                self.N_0_ph: N_0_np,
            }
        )
    
    def save_vars(self):
        self.vars_np = self.sess.run(self.vars_tf)
        return self.vars_np
    
    def load_vars(self, vals):
        for var, val in zip(self.vars_tf, vals):
            var.load(val, self.sess)
            
    def compute_sample_gradients(self, X_np):
        return self.sess.run(self.dYdX, feed_dict = {self.X_tf: X_np})[0]

    def compute_sample_lipschitz_bound(self, X_np):
        grads = self.compute_sample_gradients(X_np)
        return np.sqrt(np.square(grads).sum(1)).max()
