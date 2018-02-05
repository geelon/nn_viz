"""
LearningProblem defines a problem context in which many models can be tested.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from .learning_model import MLP
from .learning_model_utils import *

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

    def get_weights_dict(self, name):
        """
        Returns: weights, biases
        Each are a dictionary: @weights has l_1, l_2, and l_inf norms of weights 
        and @biases has the l_inf norm of biases.
        """
        assert name in self.models, "Model '{}' does not exist".format(name)

        m = self.load_model(name)
        return m.get_weights()

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
        self.models = dict()
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
            
        

