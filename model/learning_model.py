import numpy as np
import pandas as pd
import tensorflow as tf
from .learning_model_utils import *

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

        # Computation graph defined in build_graph:
        self.X = None  
        self.y = None
        self.layers = dict()
        self.loss = None
        self.train_op = None

        # Restoration information
        self.ckpt_no = 0
        self.modified = False # bit to indicate when resaving needed

        
    def build_graph(self):
        """
        Constructs computation graph, loading into memory. In general, 
        this should be called from Session._open_model(), as having multiple
        models open in the same graph will cause scoping issues.
        """
        name = self.name
        input_dim = self.input_dim
        hidden_dims = self.hidden_dims
        num_classes = self.num_classes
        
        self.X = tf.placeholder(tf.float32, shape=[None,input_dim], name=name+'X')
        self.y = tf.placeholder(tf.float32, shape=[None,num_classes], name=name+'y')
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
        

            
