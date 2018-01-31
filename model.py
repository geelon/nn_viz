import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from model_utils import *

class MLP:
    def __init__(self, name, input_dim, hidden_dims, num_classes):
        """
        @layers_dim: list of layer dimensions, including input, hidden, output
        """
        self.name = name
        self.num_layers = len(hidden_dims) + 1

        assert self.num_layers > 0, "No hidden layers"
        self.input_dim  = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes


        self.X = tf.placeholder(tf.float32, shape=[None,input_dim], name='X')
        self.y = tf.placeholder(tf.float32, shape=[None,num_classes], name='y')

        self.ckpt_no = 0
        self.modified = False # bit to indicate when resaving needed

        """
        Weights matrix variables named 'name/w_i'
        Bias variables named 'name/b_i'
        """
        layers_dim = [input_dim] + hidden_dims + [num_classes]
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            for i in range(self.num_layers):
                tf.get_variable(name="w_{}".format(i),
                                shape=[layers_dim[i], layers_dim[i+1]],
                                dtype=tf.float32)

                tf.get_variable(name="b_{}".format(i),
                                shape=[layers_dim[i+1]], dtype=tf.float32)


        """
        Computation Graph
        """
        self.layers = dict()
        layers = self.layers
        layers['layer_0'] = self.X
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            prev_layer = layers['layer_0']
            for i in range(self.num_layers):
                weight = tf.get_variable(name="w_{}".format(i))
                bias   = tf.get_variable(name="b_{}".format(i))
                curr_layer = tf.add(tf.matmul(prev_layer,weight), bias)
                layers['layer_{}'.format(i+1)] = curr_layer
                prev_layer = tf.nn.relu(curr_layer)
        self.layers = layers


        """
        Training Cycle
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

            output = layers['layer_{}'.format(self.num_layers)]
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y,
                                                           logits=output)
            loss = tf.reduce_mean(loss)
            train_op = optimizer.minimize(loss)

        self.loss = loss
        self.train_op = train_op

        """
        Visualizing

        with tf.variable_scope(name, reuse=True):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                writer = tf.summary.FileWriter("test.log", sess.graph)
                
                sess.run(layers['layer_{}'.format(self.num_layers)],
                         feed_dict={self.X: [[2,2],[1,2]]})
                
                writer.close()
        """

    def train(self):
        return self.loss, self.train_op
        


            
            
        
class LearningProblem:
    def __init__(self, input_dim, num_classes, train, test, path="/tmp/nn_log/"):
        # Problem Properties
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.train_x = train[0]
        self.train_y = train[1]
        self.test_x = test[0]
        self.test_y = test[1]

        self.num_train = self.train_y.shape[0]
        self.num_test  = self.test_y.shape[0]

        self.train_y_one_shot = to_one_shot(self.train_y, num_classes)
        self.test_y_one_shot  = to_one_shot(self.test_y , num_classes)

        # Model data
        self.models = dict()
        self.path = path

    def _create_model(self, name, model_info):
        """
        Adds a model to the problem.
        """
        hidden_dims = model_info['hidden_dims']
        save_path = model_info['save_path']
        ckpt_no = model_info['ckpt_no']
        ckpt_path = save_path + 'model.ckpt'
        
        # Creates model into memory
        m = MLP(name, self.input_dim, hidden_dims, self.num_classes)
        m.ckpt_no = ckpt_no
        m.ckpt_path = ckpt_path

        
        # Creates location to save model
        try:
            os.makedirs(save_path)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise

        # Records information about model
        record_model(save_path + "info", model_info)
          
        self.models[name] = m
        return m

    def _get_model_info(self, name, hidden_dims=None):
        """
        Returns dict of information about model.
        """
        # Get path for model
        save_path = get_model_path(self.path, name)

        # Model does not exist yet
        if name not in self.models:
            assert hidden_dims is not None, "Need to define hidden_dims"
            return { 'name':name,
                     'hidden_dims':hidden_dims,
                     'save_path':save_path,
                     'ckpt_no':0 }
        
        m = self.models[name]
        # Model is not in memory
        if isinstance(m, str):
            return get_model(m)

        # Model is in memory
        model_info = { 'name':m.name,
                       'hidden_dims':m.hidden_dims,
                       'save_path':get_model_path(self.path,name),
                       'ckpt_no':m.ckpt_no }
        return model_info


  
    def load_model(self, name, hidden_dims=None):
        """
        Loads model into memory. If model does not exists, creates it,
        in which case hidden_dims must not be None.
        """
        if name in self.models:
            m = self.models[name]
            if not isinstance(m, str):
                return m
        model_info = self._get_model_info(name, hidden_dims)
        self.models[name] = self._create_model(name, model_info)
        return self.models[name]
        
            


    def pop_model(self, name):
        """
        Pops model out of memory, replacing value in dictionary with
        path to saved path.
        """
        # Model exists
        assert name in self.models, "Model '{}' does not exist".format(name)

        model_info = self._get_model_info(name)
        m = self.models.pop(name)

        # Save model to file
        save_path = model_info['save_path'] + 'info'
        record_model(save_path, model_info)

        # Leave path to saved model
        self.models[name] = save_path

    def get_layers_dict(self, name):
        """
        Returns dictionary of layers in model. Assumes model exists.
        """
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        return m.layers

    def get_train_dict(self, name):
        """
        Returns dictionary with 'loss' and 'train_op' in model. Assumes exists.
        """
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        loss, train_op = m.train()
        return {'loss':loss, 'train_op':train_op}

    def get_feed_names(self, name):
        """
        Returns uninitialized placeholders.
        """
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        return (m.X, m.y)

    def get_feed_train(self, name):
        """
        Returns feed_dict for training.
        """
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        return { m.X : self.train_x, m.y : self.train_y_one_shot }

    def get_feed_test(self, name):
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        return { m.X : self.test_x, m.y : self.test_y_one_shot }

    def clean_up(self):
        """
        Removes all files and directories generated.
        """
        path = self.path
        answer = input("Remove all subfiles of '{}'? [y/N]".format(path))
        if answer is "y" or answer is "Y":
            try:
                shutil.rmtree(path)
                print("Files removed.")
            except OSError as e:
                if e.errno != os.errno.ENOENT:
                    raise
        else:
            print("Files not removed.")
            
        



        
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
        self.curr_model = self.learning_problem.load_model(name,hidden_dims)
        
        # Begin session, restoring variables if necessary
        self.curr_session = tf.Session()
        ckpt_path = self.curr_model.ckpt_path
        if os.path.isfile(ckpt_path) and os.stat(ckpt_path).st_size > 0:
            print("Restoring variables")
            tf.train.Saver().restore(self.curr_session, ckpt_path)


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

        # Closes session
        self.curr_session.close()

        # Stow from memory into disk
        self.learning_problem.models[name] = self.curr_model
        self.learning_problem.pop_model(name)
        self.curr_model = None
        self.curr_session = None
        
        
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
