import numpy as np
import tensorflow as tf
from lib.neuralnetwork import Fair_MNN
from lib.zemel import ZemelFairRepresentations, k_nearest_neighbors_sp
from lib.general_functions import now
from lib.metrics import accuracy_np, discrimination_np, consistency_np
from lib.metrics import k_nearest_neighbors_sp, identify_monotonic_pairs
from lib.metrics import resentment_individual, resentment_pairwise
from lib.metrics import lipschitz_sample_estimate
from matplotlib import pyplot as plt
import pickle as pk
from lib.compas import load_data as load_data_compas
from lib.law_school import load_data as load_data_law_school
from lib.german import load_data as load_data_german

for data_name in ['compas', 'law_school', 'german']:
    print now(), data_name
    tf.reset_default_graph()
    if data_name == 'compas':
        load_data = load_data_compas
        monotonicity = [1, 0, 1, 1, 1]
        data = load_data(binarize=True)

    if data_name == 'law_school':
        load_data = load_data_law_school
        monotonicity = [1, 1]
        data = load_data(binarize=True)

    if data_name == 'german':
        load_data = load_data_german
        data = load_data()
        monotonicity = data['monotonicity']

    use_minibatch = True

    valid_frac = 0.2
    n_valid = int(data['data_train'].shape[0] * valid_frac)

    X_train = data['data_train'][:-n_valid, data['X_cols']].astype(np.number)
    X_valid = data['data_train'][-n_valid:, data['X_cols']].astype(np.number)
    X_test  = data['data_test' ][:, data['X_cols']].astype(np.number)
    Y_train = data['data_train'][:-n_valid, data['Y_col']:data['Y_col']+1].astype(np.number)
    Y_valid = data['data_train'][-n_valid:, data['Y_col']:data['Y_col']+1].astype(np.number)
    Y_test  = data['data_test' ][:, data['Y_col']:data['Y_col']+1].astype(np.number)
    A_train = data['data_train'][:-n_valid, data['A_cols']][:, -2:].astype(np.number)
    A_valid = data['data_train'][-n_valid:, data['A_cols']][:, -2:].astype(np.number)
    A_test  = data['data_test' ][:, data['A_cols']][:, -2:].astype(np.number)
    n, x_dim = X_train.shape

    X_pretrain = X_train + 0.
    Y_prevalid = X_valid + 0.
    X_pretest  = X_test + 0.

    std = X_train.std(axis=0)
    X_train = X_train / std
    X_valid = X_valid / std
    X_test  = X_test  / std

    if use_minibatch:
        batch_size = 128 if data_name == 'german' else 256
        X_tf, Y_tf, A_tf = [tf.constant(d) for d in [X_train, Y_train, A_train]]
        ds = tf.data.Dataset.from_tensor_slices({'X': X_tf, 'Y': Y_tf, 'A': A_tf})
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=batch_size * 3)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.prefetch(batch_size)
        iterator = ds.make_initializable_iterator()
        next_element = iterator.get_next()
        X_trn, Y_trn, A_trn = [next_element[key] for key in ['X', 'Y', 'A']]
    else:
        batch_size = X_train.shape[0]
        X_trn, Y_trn, A_trn = [tf.constant(d) for d in [X_train, Y_train, A_train]]

    batch_factor = n / batch_size
    n, x_dim = X_train.shape

    fnn = Fair_MNN(
        x_dim = x_dim,
        a_dim = 2,
        var_scope = 'fair_mnn',
        nn_params = {
            'dtype': tf.float64,
            'num_layers': 5, 
            'width': [[10], [10], [10], [10], [1]], 
            'output_dim': 1, 
            'activations': [tf.tanh], 
            'activate_last_layer': False, 
            'monotonicity': [0] * x_dim, # None = all positive
            'positive_func': lambda w: 1. + tf.nn.elu(w - 1.),
            'var_scope': 'ffnn',
        },
        fairness_loss_type = ['DP', 'EOdds', 'EOutcome', 'EOpportunity', 'ZemelDiscrimination'][4],
        y_loss_type = ['CE', 'EC'][0],
        use_minibatch = False,
        batch_size = batch_size,
        X_minibatch_tf = X_trn if use_minibatch else None,
        Y_minibatch_tf = Y_trn if use_minibatch else None,
        A_minibatch_tf = A_trn if use_minibatch else None,
    )
    
    fmnn = Fair_MNN(
        x_dim = x_dim,
        a_dim = 2,
        var_scope = 'fair_mnn',
        nn_params = {
            'dtype': tf.float64,
            'num_layers': 5, 
            'width': [[10], [10], [10], [10], [1]], 
            'output_dim': 1, 
            'activations': [tf.tanh], 
            'activate_last_layer': False, 
            'monotonicity': monotonicity, # None = all positive
            'positive_func': lambda w: 1. + tf.nn.elu(w - 1.),
            'var_scope': 'mffnn',
        },
        fairness_loss_type = ['DP', 'EOdds', 'EOutcome', 'EOpportunity', 'ZemelDiscrimination'][4],
        y_loss_type = ['CE', 'EC'][0],
        use_minibatch = False,
        batch_size = batch_size,
        X_minibatch_tf = X_trn if use_minibatch else None,
        Y_minibatch_tf = Y_trn if use_minibatch else None,
        A_minibatch_tf = A_trn if use_minibatch else None,
    )

    num_clusters = 10
    zfr = ZemelFairRepresentations(
        x_dim = x_dim,
        num_clusters = num_clusters,
        sigmas = X_train.var(axis=0),
        var_scope = 'ZFR',
        dtype=tf.float64,
        optimizer=tf.train.AdamOptimizer,
    )

    sess = tf.Session()
    fnn.sess=sess
    fnn.sess=sess
    zfr.sess=sess
    if use_minibatch:
        sess.run(iterator.initializer)

        
    n_runs = 250
    n_steps = 30
    n_runs = 100
    n_steps = 30
    fnn_alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

    fnn_loss_by_step = np.zeros([n_runs, n_steps])
    fnn_losses = {}
    fnn_states = {}

    print ''
    print now(), 'fnn'
    for run in range(n_runs):
        if run < len(fnn_alphas):
            alpha = fnn_alphas[run] + 0.
        else:
            alpha = np.random.beta(.5, .5, 1)[0]
        fnn_losses[alpha, run] = np.inf
        fnn_states[alpha, run] = []
        
        fnn.initialize(sess=sess)

        # Initialize the bias semi-intelligently
        fnn.vars_tf[-1].load(np.array([0.]), sess)

        for step in range(n_steps):
            losses = fnn.fit(
                X_np = None if use_minibatch else X_train, 
                Y_np = None if use_minibatch else Y_train, 
                A_np = None if use_minibatch else A_train,
                A_0 = [0.4, 0.6],
                N_0 = 0.5,
                lambda_A = alpha, 
                lambda_N = 0., 
                lambda_Y = 1. - alpha,
                n_steps = int(10 * batch_factor),
                learning_rate = (
                    1e-2 / batch_factor if step < 33 else 
                    1e-3 / batch_factor # if step < 100 else
                ),
                sess = sess,
            )
            losses = fnn.predict_and_losses(
                X_np = X_valid, 
                Y_np = Y_valid, 
                A_np = A_valid,
                A_0_np = [Y_train.mean()]*2,
                N_0_np = Y_train.mean(),
                lambda_A = alpha, 
                lambda_N = 0., 
                lambda_Y = 1. - alpha,
            )[1]
            fnn_loss_by_step[run, step] = losses['loss'] + 0.
            if losses['loss'] < fnn_losses[alpha, run]:
                fnn_losses[alpha, run] = losses['loss'] + 0.
                fnn_states[alpha, run] = [v + 0. for v in fnn.save_vars()]

        print now(), run, alpha, losses['loss']

    pk.dump(
        {
            'fnn_losses': fnn_losses, 
            'fnn_states': fnn_states, 
            'fnn_loss_by_step': fnn_loss_by_step,
        }, 
        open('data/{:}_fnn_var_state_3.pk'.format(data_name), 'wb'),
    )
    
    n_runs = 100
    n_steps = 30
    fmnn_alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]

    fmnn_loss_by_step = np.zeros([n_runs, n_steps])
    fmnn_losses = {}
    fmnn_states = {}

    print ''
    print now(), 'fmnn'
    for run in range(n_runs):
        if run < len(fmnn_alphas):
            alpha = fmnn_alphas[run] + 0.
        else:
            alpha = np.random.beta(.5, .5, 1)[0]
        fmnn_losses[alpha, run] = np.inf
        fmnn_states[alpha, run] = []
        
        fmnn.initialize(sess=sess)

        # Initialize the bias semi-intelligently
        fmnn.vars_tf[-1].load(np.array([0.]), sess)

        for step in range(n_steps):
            losses = fmnn.fit(
                X_np = None if use_minibatch else X_train, 
                Y_np = None if use_minibatch else Y_train, 
                A_np = None if use_minibatch else A_train,
                A_0 = [0.4, 0.6],
                N_0 = 0.5,
                lambda_A = alpha, 
                lambda_N = 0., 
                lambda_Y = 1. - alpha,
                n_steps = int(10 * batch_factor),
                learning_rate = (
                    1e-2 / batch_factor if step < 33 else 
                    1e-3 / batch_factor # if step < 100 else
                ),
                sess = sess,
            )
            losses = fmnn.predict_and_losses(
                X_np = X_valid, 
                Y_np = Y_valid, 
                A_np = A_valid,
                A_0_np = [Y_train.mean()]*2,
                N_0_np = Y_train.mean(),
                lambda_A = alpha, 
                lambda_N = 0., 
                lambda_Y = 1. - alpha,
            )[1]
            fmnn_loss_by_step[run, step] = losses['loss']
            if losses['loss'] < fmnn_losses[alpha, run]:
                fmnn_losses[alpha, run] = losses['loss'] + 0.
                fmnn_states[alpha, run] = [v + 0. for v in fmnn.save_vars()]

        print now(), run, alpha, losses['loss']

    pk.dump(
        {
            'fmnn_losses': fmnn_losses, 
            'fmnn_states': fmnn_states, 
            'fmnn_loss_by_step': fmnn_loss_by_step,
        }, 
        open('data/{:}_fmnn_var_state_3.pk'.format(data_name), 'wb'),
    )

    n_runs = 100
    n_steps = 40
    zfr_alphas = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    zfr_lambdas_X = 10. ** np.array([-1., -.5, 0., .5, 1.])

    zfr_loss_by_step = np.zeros([n_runs, n_steps])
    zfr_losses = {}
    zfr_states = {}
    
    print ''
    print now(), 'zfr'

    for run in range(n_runs):
        if run < len(zfr_alphas) * len(zfr_lambdas_X):
            alpha = zfr_alphas[run % len(zfr_alphas)] + 0.
            lambda_X = zfr_lambdas_X[run // len(zfr_alphas)] + 0.
        else:
            alpha = np.random.beta(.5, .5, 1)[0]
            lambda_X = 10 ** (2. * np.random.beta(1., 1., 1)[0] - 1.)
        
        zfr_losses[alpha, lambda_X, run] = np.inf
        zfr_states[alpha, lambda_X, run] = []
        
        zfr.initialize(sess=sess)
        
        # Do a smart(er) initialization
        X_sample = X_train[np.random.choice(X_train.shape[0], size=num_clusters), :]
        while np.array([X_sample[i, :] == X_sample[j, :] 
                        for i in range(num_clusters-1) 
                        for j in range(i+1, num_clusters)]
                      ).min(1).max():
            X_sample = X_train[np.random.choice(X_train.shape[0], size=num_clusters), :]
        zfr.set_weights(
            V_np = X_sample.transpose(),
            W_np = np.random.uniform(size=[num_clusters, 1]),
        )

        for step in range(n_steps):
            zfr.fit(
                X_np = X_train, 
                Y_np = Y_train, 
                A_np = A_train,
                alpha_Z = alpha, 
                alpha_X = lambda_X, 
                alpha_Y = 1. - alpha,
                n_steps = n_steps,
                learning_rate = 1e-2 if step < 30 else 1e-3,
                sess=sess,
            )
            loss = zfr.predict_and_losses(
                X_np = X_valid, 
                Y_np = Y_valid, 
                A_np = A_valid, 
                alpha_Z = alpha, 
                alpha_X = 0., 
                alpha_Y = 1. - alpha,
            )[1][0]
            zfr_loss_by_step[run, step] = loss + 0.
        zfr_losses[alpha, lambda_X, run] = loss + 0.
        zfr_states[alpha, lambda_X, run] = [zfr.W_np + 0., zfr.V_np + 0.]
        print now(), run, alpha, lambda_X, loss

    pk.dump(
        {
            'zfr_losses': zfr_losses, 
            'zfr_states': zfr_states, 
            'zfr_loss_by_step': zfr_loss_by_step,
        }, 
        open('data/{:}_zfr_var_state_3.pk'.format(data_name), 'wb'),
    )
    
    print ''
    sess.close()
