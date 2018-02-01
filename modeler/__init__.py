import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
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
        

    def quick_train(self, epochs=1000):
        """
        Run training on current model. Saves loss.
        """
        assert self.curr_model is not None, "Model not specified for training."

        tf_objs = self.model_properties()
        fetches = [tf_objs['loss'], tf_objs['train_op']]
        feed_dict = self.model_feed_train()

        loss_curve = pd.DataFrame(0.0, index=np.arange(epochs), columns=['loss'])
        
        for i in range(epochs):
            loss_curve['loss'][i] , _ = self.run(fetches, feed_dict)

        # Save loss curve to save_path/name/loss_curve_{ckpt_no}.csv
        save_path = self.learning_problem.path
        name = self.curr_model.name
        ckpt_no = self.curr_model.ckpt_no
        loss_curve.to_csv(save_path + name + '/loss_curve_{}'.format(ckpt_no))

        return loss_curve

        
def nn_stats(mlp, X, y, epochs=1000):
    """
    
    """
    loss_curve = pd.DataFrame(0.0, index=np.arange(epochs), columns=['loss'])
    feed_dict = {mlp.X : X, mlp.y : y}

    with tf.variable_scope(mlp.name, reuse=True):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(epochs):
                loss, train = mlp.train()
                lo, _ = sess.run([loss,train], feed_dict=feed_dict)
                loss_curve['loss'][i] = lo
    return loss_curve
