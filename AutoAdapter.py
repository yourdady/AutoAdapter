''' 
@project AutoAdapter
@author Peng
@file AutoAdapter.py
@time 2018-08-16
'''
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics.scorer import accuracy_score
MODEL_SAVE_PATH = './model'
MODEL_NAME = 'model.ckpt'

class autoAdapter():
    def __init__(self, input_dim, new_dim, n_classes, model_path = None, lamb = 0.01, learning_rate = 0.01
                 , batch_size_src = 128, batch_size_tar = 128, training_steps = 5000, l2 = 0.001,
                 optimizer = 'GD', save_step = 20, print_step = 20, kernel_type = 'linear', sigma_list=None,
                 ** kernel_param):
        """
        
        :param input_dim: 
        :param new_dim: 
        :param n_classes: 
        :param model_path: 
        :param lamb: 
        :param learning_rate: 
        :param batch_size_src: 
        :param batch_size_tar: 
        :param training_steps: 
        :param l2: 
        :param optimizer: 
        :param save_step: 
        :param print_step: 
        :param kernel_type: 
        :param kernel_param: 
        """
        self.model_path = model_path
        self.input_dim = input_dim
        self.new_dim = new_dim
        self.n_classes = n_classes
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.batch_size_src = batch_size_src
        self.batch_size_tar = batch_size_tar
        self.training_steps = training_steps
        self.l2 = l2
        self.optimizer = optimizer
        self.save_step = save_step
        self.print_step = print_step
        self.kernel_type = kernel_type
        self.kernel_param = kernel_param
        self.sigma_list = sigma_list
    def dann(self, Xsrc=None, Xtar=None):
        """
        
        :param Xsrc: 
        :param Xtar: 
        :return: 
        """
        if Xtar is None:
            raise TypeError("Xtar is None??")
        def fully_connected(input_layer, weights, biases):
            return tf.add(tf.matmul(input_layer, weights), biases)

        with tf.name_scope("layer1"):
            weights_1 = tf.get_variable("weights_1",shape = [self.input_dim, self.new_dim],
                                        initializer=tf.truncated_normal_initializer(stddev=1))
            bias_1 = tf.get_variable("bias_1", shape = [self.new_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=1))
            hidden_layer_src = None
            if Xsrc is not None:
                hidden_layer_src = tf.nn.relu(fully_connected(Xsrc, weights_1, bias_1))
            hidden_layer_tar = tf.nn.relu(fully_connected(Xtar, weights_1, bias_1))

        with tf.name_scope("layer2"):
            weights_2 = tf.get_variable("weights_2",shape = [self.new_dim, self.n_classes],
                                        initializer=tf.truncated_normal_initializer(stddev=1))
            bias_2 = tf.get_variable("bias_2", shape = [self.n_classes],
                                     initializer=tf.truncated_normal_initializer(stddev=1))
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(self.l2)(weights_2))
            logits = None
            if Xsrc is not None:
                logits = fully_connected(hidden_layer_src, weights_2, bias_2)

        global saver
        saver = tf.train.Saver([
            weights_1,
            bias_1,
            weights_2,
            bias_2
        ])
        return logits, hidden_layer_src, hidden_layer_tar


    def linear_mmd2(self, f_of_X_src, f_of_X_tar):
        """
        
        :param f_of_X: 
        :param f_of_Y: 
        :return: 
        """
        delta = f_of_X_src - f_of_X_tar
        loss = tf.reduce_mean(tf.matmul(delta, tf.transpose(delta)))
        return loss

    def poly_mmd2(self, f_of_X, f_of_Y, kernel_param):
        """
        
        :param f_of_X: 
        :param f_of_Y: 
        :param d: 
        :param alpha: 
        :param c: 
        :return: 
        """
        try:
            alpha = kernel_param["alpha"]
        except KeyError:
            alpha = 1.0
        try:
            d = kernel_param["d"]
        except KeyError:
            d = 1
        try:
            c = kernel_param["c"]
        except KeyError:
            c = 2.0

        K_XX = (tf.reduce_sum(alpha * (f_of_X[:-1] * f_of_X[1:]), 1) + c)
        K_XX_mean = tf.reduce_mean(tf.pow(K_XX,d))

        K_YY = (tf.reduce_sum(alpha * (f_of_Y[:-1] * f_of_Y[1:]), 1) + c)
        K_YY_mean = tf.reduce_mean(tf.pow(K_YY,d))

        K_XY = (tf.reduce_sum(alpha * (f_of_X[:-1] * f_of_Y[1:]), 1) + c)
        K_XY_mean = tf.reduce_mean(tf.pow(K_XY,d))

        K_YX = (tf.reduce_sum(alpha * (f_of_Y[:-1] * f_of_X[1:]), 1) + c)
        K_YX_mean = tf.reduce_mean(tf.pow(K_YX,d))

        loss = K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean
        return loss


    def mix_rbf_mmd2(self, X_src, X_tar, sigma_list, biased=True):
        """
        
        :param X_src: 
        :param X_tar: 
        :param sigma_list: 
        :param biased: 
        :return: 
        """
        # K_SS, K_ST, K_TT, d
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X_src, X_tar, sigma_list)

        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)

    def mix_rbf_mm2_ratio(self):
        pass

    def mmd(self, X_src, X_tar):
        X_src_mean = np.mean(X_src, axis=0)
        X_tar_mean = np.mean(X_tar, axis=0)
        mmd = np.sum((X_src_mean-X_tar_mean)**2)
        return mmd

    def fit(self, data_src, data_tar, onehot=False, plot = False):
        """
        
        :param data_src: 
        :param data_tar: 
        :param onehot: 
        :return: 
        """
        with tf.Graph().as_default() as g:
            global_step = tf.Variable(0, trainable=False)
            X_src_placeholder = tf.placeholder(shape=[self.batch_size_src, self.input_dim], dtype=tf.float32, name='Xsrc')
            X_tar_placeholder = tf.placeholder(shape=[self.batch_size_tar, self.input_dim], dtype=tf.float32, name='Xtar')
            y_placeholder = tf.placeholder(tf.float32, [None, self.n_classes], name='y-input')

            output, hidden_src, hidden_tar = self.dann(X_src_placeholder, X_tar_placeholder)

            loss_y = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_placeholder)
            if self.kernel_type == 'linear':
                loss_mmd = self.linear_mmd2(hidden_src, hidden_tar)
            if self.kernel_type == 'poly':
                loss_mmd = self.poly_mmd2(hidden_src, hidden_tar, self.kernel_param)
            if self.kernel_type == 'rbf':
                if self.sigma_list is None:
                    raise ValueError("sigma list is None??")
                loss_mmd = self.mix_rbf_mmd2(hidden_src, hidden_tar, sigma_list=self.sigma_list)
            else:
                loss_mmd = tf.constant(0, dtype='float')
            loss_y_mean = tf.reduce_mean(loss_y)
            loss = tf.constant(self.lamb) * loss_mmd + loss_y_mean + tf.add_n(tf.get_collection('losses'))

            if self.optimizer == 'GD':
                train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step=global_step)
                grad = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(loss)

            else:
                train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss,global_step=global_step)
                grad = tf.train.GradientDescentOptimizer(self.learning_rate).compute_gradients(loss)
            with tf.control_dependencies([train_step]):
                train_op = tf.no_op(name='train')

            saver = tf.train.Saver()
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                for i in range(self.training_steps):
                    xs, ys = data_src.dataset.train.next_batch(self.batch_size_src)
                    xt, _ = data_tar.dataset.train.next_batch(self.batch_size_tar)

                    xs = np.reshape(xs, (self.batch_size_src, self.input_dim))
                    xt = np.reshape(xt, (self.batch_size_tar, self.input_dim))

                    if onehot == False:
                        ys = self._transformlabel(ys, self.n_classes)
                    _, loss_, loss_y_mean_, loss_mmd_, output_, hidden_src_, hidden_tar_ = sess.run([train_step, loss, loss_y_mean, loss_mmd, output,
                                                                           hidden_src, hidden_tar],
                                                              feed_dict={
                                                                  X_src_placeholder: xs,
                                                                  X_tar_placeholder: xt,
                                                                  y_placeholder: ys
                                                              })
                    acc = accuracy_score(np.argmax(output_, 1),
                                         np.argmax(ys, 1))
                    if i%self.print_step == 0:
                        print("After {} training steps\n loss_mmd on training batch is:{}"
                              "\n loss_y on training batch is:{}"
                              "\n loss on training batch is:{}".format(i, loss_mmd_, loss_y_mean_, loss_))
                        print("accuracy on train set:{}".format(acc))
                        print("mmd: ", self.mmd(hidden_src_, hidden_tar_))
                    if i%self.save_step == 0:
                        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

    def transform(self, X):
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [
                len(X),
                self.input_dim],name='x-input')

            _,_,X_new = self.dann(Xsrc=None, Xtar=X)
            saver = tf.train.Saver()

            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    X_new = sess.run([X_new], feed_dict={x: X})
                    print("X_new : ", X_new[0])
                    print("shape : ", np.array(X_new).shape)
                    return X_new
                else:
                    print("No checkpoint file found")
                    raise FileNotFoundError



    ###############################################################################
    # Helper functions to compute variances based on kernel matrices
    ################################################################################

    def _mmd2(self, K_XX, K_XY, K_YY, const_diagonal = False, biased = False):
        """
        
        :param K_XX: 
        :param K_XY: 
        :param K_YY: 
        :param const_diagonal: 
        :param biased: 
        :return: 
        """
        m = K_XX.shape[0]
        if const_diagonal is not False:
            diag_X = diag_Y = const_diagonal
            sum_diag_X = sum_diag_Y = m * const_diagonal
        else:
            diag_X = tf.diag(K_XX)  # (m,)
            diag_Y = tf.diag(K_YY)  # (m,)
            sum_diag_X = tf.reduce_sum(diag_X)
            sum_diag_Y = tf.reduce_sum(diag_Y)

        Kt_XX_sums = tf.reduce_sum(K_XX, axis=1) - diag_X  # \tilde{K}_XX * e = K_XX * e - diag_X
        Kt_YY_sums = tf.reduce_sum(K_YY, axis=1) - diag_Y  # \tilde{K}_YY * e = K_YY * e - diag_Y
        K_XY_sums_0 = tf.reduce_sum(K_XY, axis=0)  # K_{XY}^T * e

        Kt_XX_sum = tf.reduce_sum(Kt_XX_sums)  # e^T * \tilde{K}_XX * e
        Kt_YY_sum = tf.reduce_sum(Kt_YY_sums)  # e^T * \tilde{K}_YY * e
        K_XY_sum = tf.reduce_sum(K_XY_sums_0)  # e^T * K_{XY} * e

        if biased:
            mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                    + (Kt_YY_sum + sum_diag_Y) / (m * m)
                    - 2.0 * K_XY_sum / (m * m))
        else:
            mmd2 = (Kt_XX_sum / (m * (m - 1))
                    + Kt_YY_sum / (m * (m - 1))
                    - 2.0 * K_XY_sum / (m * m))

        return mmd2



    def _mix_rbf_kernel(self, X_src, X_tar, sigma_list):
        """
        
        :param X_src: 
        :param X_tar: 
        :param sigma_list: 
        :return: 
        """
        assert (X_src.shape[0] == X_tar.shape[0])
        m = X_src.shape[0]
        Z = tf.concat((X_src, X_tar), 0)
        ZZT = tf.matmul(Z, tf.transpose(Z))
        diag_ZZT = tf.diag(ZZT)

        Z_norm_sqr = diag_ZZT.expand_as(ZZT)
        exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr

        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2 * sigma ** 2)
            K += tf.exp(-gamma * exponent)

        return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


    def _mmd2_and_variance(self):

        pass

    def _transformlabel(self, y, n_classes):
        """
        
        :param y: 
        :param n_classes: 
        :return: 
        """
        label_ = np.zeros([len(y), n_classes])
        for i in range(len(y)):
            label_[i][y[i]] = 1
        return label_