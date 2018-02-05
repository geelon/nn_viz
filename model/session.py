"""
Session enables quickly switching and saving models.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
import shutil
from .session_utils import empty_df
from .learning_model import MLP
from .learning_problem import LearningProblem



        
class Session:
    def __init__(self, learning_problem):
        self.learning_problem = learning_problem
        self.curr_model = None


    def _open_model(self, name, hidden_dims=None):
        """
        Unsafe open model; assumes no existing model unsaved.
        """
        sess = self.curr_session
        if sess is not None:
            sess.close()

        # Retrieve model
        m = self.learning_problem.load_model(name,hidden_dims)
        m.build_graph()
        self.curr_model = m
        
        # Begin session, restoring variables if necessary
        self.curr_session = tf.Session()
        ckpt_path = self.curr_model.ckpt_path
        if os.path.isfile(ckpt_path + ".index"):
            print("Restoring variables for {}.".format(name))
            tf.train.Saver().restore(self.curr_session, ckpt_path)
        else:
            print("Initializing variables for {}.".format(name))
            self.curr_session.run(tf.global_variables_initializer())
            


    def _close_model(self):
        """
        Closes current model, saving out.
        """
        name = self.curr_model.name
        
        # Saves model
        if self.curr_model.modified:
            ckpt_path = self.curr_model.ckpt_path
            saver = tf.train.Saver()
            saver.save(self.curr_session, ckpt_path)
            self.curr_model.ckpt_no += 1
            self.curr_model.modified = False

        # Closes session
        self.curr_session.close()

        # Stow from memory into disk
        self.learning_problem.models[name] = self.curr_model
        self.learning_problem.pop_model(name)
        self.curr_model = None
        self.curr_session = None

        # Reset computation graph
        tf.reset_default_graph()
        
    def __enter__(self):
        self.curr_session = tf.Session()
        return self

    def __exit__(self, exeception_type, exception_value, traceback):
        if self.curr_model is not None:
            self._close_model()
        if self.curr_session is not None:
            self.curr_session.close()

        
    def set_current_model(self, name):
        """
        Set model to run in current session.
        """
        if self.curr_model is not None:
            self._close_model()
            
        return self._open_model(name)
        
    
        
    def run(self, *args, **kwargs):
        """
        Pass args to tf.Session.run(args).
        """
        if self.curr_model is not None:
            self.curr_model.ckpt_no += 1
        self.curr_model.modified = True
        return self.curr_session.run(*args, **kwargs)

    def model_properties(self, verbose=False):
        """
        Returns useful objects within model.
        """
        if self.curr_model is None:
            print("No model loaded into session.")
            return

        name = self.curr_model.name
        layers_dict = self.learning_problem.get_layers_dict(name)
        train_dict = self.learning_problem.get_train_dict(name)
        tf_objs = {**layers_dict, **train_dict}
        if verbose:
            print("The keys are {}".format(tf_objs.keys()))
        return tf_objs

    def model_feed_train(self, verbose=False):
        """
        Returns variables for feed dict in model.
        """
        if self.curr_model is None:
            print("No model loaded into session.")
            return

        name = self.curr_model.name
        feed_dict = self.learning_problem.get_feed_train(name)
        if verbose:
            print("The keys are {}".format(feed_dict.keys()))
        return feed_dict
        
    def model_feed_test(self, verbose=False):
        """
        Returns variables for feed dict in model.
        """
        if self.curr_model is None:
            print("No model loaded into session.")
            return

        name = self.curr_model.name
        feed_dict = self.learning_problem.get_feed_test(name)
        if verbose:
            print("The keys are {}".format(feed_dict.keys()))
        return feed_dict
    
    def quick_train(self, epochs=1000, validate=False):
        """
        Run training on current model. Saves loss.
        """
        assert self.curr_model is not None, "Model not specified for training."

        tf_objs = self.model_properties()
        fetches = [tf_objs['loss'], tf_objs['train_op']]
        feed_dict = self.model_feed_train()

        loss_curve = pd.DataFrame(0.0, index=np.arange(epochs), columns=['loss'])
        
        if validate:
            feed_dict_v = self.model_feed_test()
            loss_curve_v = pd.DataFrame(0.0, index=np.arange(epochs),
                                        columns=['loss_test'])

        
        for i in range(epochs):
            loss_curve['loss'][i] , _ = self.run(fetches, feed_dict)
            if validate:
                loss_curve_v['loss_test'][i] = self.run(tf_objs['loss'],
                                                        feed_dict=feed_dict_v)

        # Save loss curve to save_path/name/loss_curve_{ckpt_no}.csv
        save_path = self.learning_problem.path
        name = self.curr_model.name
        ckpt_no = self.curr_model.ckpt_no
        path = save_path + name
        loss_curve.to_csv(path + '/loss_curve_{}'.format(ckpt_no))
        if validate:
            loss_curve_v.to_csv(path + '/test_loss_curve_{}'.format(ckpt_no))
            return loss_curve, loss_curve_v
        
        return loss_curve

    def all_stats(self, epochs=1000, validate=False, verbose=True):
        """
        Run training on current model. Saves loss and weights.
        """
        assert self.curr_model is not None, "Model is not specified for training."
        name = self.curr_model.name
        num_layers = self.curr_model.num_layers

        tf_objs = self.model_properties()
        train_op = tf_objs['train_op']
        loss = tf_objs['loss']
        weights, biases = self.learning_problem.get_weights_dict(name)
        
        fetches = [train_op, loss]
        feed_dict = self.model_feed_train()

        loss_curve = empty_df(epochs, columns=['loss'])
        weights_columns = ['w_{}'.format(i) for i in range(num_layers)]
        weights_curve_l1 = empty_df(epochs, columns=weights_columns)
        weights_curve_l2 = empty_df(epochs, columns=weights_columns)
        weights_curve_l_inf = empty_df(epochs, columns=weights_columns)
        
        biases_curve = empty_df(epochs, columns=['bias_l_inf'])

        
        if validate:
            feed_dict_v = self.model_feed_test()
            loss_curve_v = empty_df(epochs, columns=['loss_test'])

        for i in range(epochs):
            if verbose:
                sys.stdout.write('\rCurrent Epoch: {}'.format(i))
                sys.stdout.flush()
            _, loss_curve['loss'][i] = self.run(fetches, feed_dict)
            weights_curve_l2.iloc[i] = self.run(weights[1], feed_dict)
            biases_curve['bias_l_inf'][i] = self.run(biases, feed_dict)
            
            if validate:
                loss_curve_v['loss_test'][i] = self.run(tf_objs['loss'],
                                                        feed_dict=feed_dict_v)
        # Save loss curve to save_path/name/loss_curve_{ckpt_no}.csv
        save_path = self.learning_problem.path
        name = self.curr_model.name
        ckpt_no = self.curr_model.ckpt_no
        path = save_path + name
        loss_curve.to_csv(path + '/loss_curve_{}'.format(ckpt_no))
        weights_curve_l1.to_csv(path + '/weights_curve_l1_{}'.format(ckpt_no))
        biases_curve.to_csv(path + '/biases_curve_{}'.format(ckpt_no))
        if validate:
            loss_curve_v.to_csv(path + '/test_loss_curve_{}'.format(ckpt_no))
            return loss_curve, loss_curve_v, weights_curve_l2, biases_curve
        
        return loss_curve, weights_curve_l2, biases_curve
        
        

