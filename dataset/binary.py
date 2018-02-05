"""
Converts datasets into weighted binary-values. This separates partitions the
range of the dataset; assigns each datapoint to a class with confidence score
between 0 and 1. Note there are many reasonable ways to assign such a confidence
score.

Ideally, a learning algorithm could determine a representation that maximizes
entropy; however, for now, we could just either split up the range into equal
intervals (i.e. collection of means and equal variances), or perhaps use k-means
to determine select the means.
"""
from math import exp
import pandas as pd
import numpy as np


class FeatureFamily:
    """
    Contains a family of n functions from input space X to [0,1], representing
    the probability that the input exhibits the feature.
    """
    def __init__(self, name):
        self.features = dict()
        self.name = name

    def add_feature(self, feature, name=None):
        """
        @feature should be a function from X to [0,1].
        @name should be a string; default is self.name/feature_number
        """
        if name is None:
            name = self.name + '/{}'.format(len(self.features))

        self.features[name] = feature

    def remove_feature(self, name):
        try:
            self.features.pop(name)
        except:
            raise Exception('Removing feature {} failed.'.format(name))

    def transform_data(self, data):
        """
        If features are a list f_i and data is x_j, returns the array f_i(x_j).
        In the future, this can be made faster using concurrency.
        """
        feature_names = self.features.keys()
        output = [[f(x) for _, f in self.features.items()] for x in data.values]
        return pd.DataFrame(output, columns=feature_names)

def gmm(x, mean, variance):
    out = max(0, 1 - (x - mean)**2 / variance)
    return out
        
            
        
def generate_gaussian_mixture(means, variances, name=None):
    """
    Given a list of means and variances,
    """
    if name is None:
        name = 'GMM'
    feature_family = FeatureFamily(name)
    for i in range(len(means)):
        feature_family.add_feature((lambda y: (lambda x: gmm(x,means[y],variances[y])))(i))
    return feature_family


def generate_equal_intervals(data, n_bins, name=None):
    """
    Generate n_bins number of features spaced evenly across the range of the
    data.
    """
    maximum = data.values.max()
    minimum = data.values.min()
    step_size = (maximum - minimum) / n_bins
    init = minimum + step_size / 2

    means = [ init + i * step_size for i in range(n_bins)]
    variances = [ step_size**2 for i in range(n_bins)]

    return generate_gaussian_mixture(means, variances, name)


class DataTransform:
    def __init__(self, sample_data, n_bins):
        self.num_class = len(sample_data.keys())
        self.transforms = dict()

        for column in sample_data:
            col_vals = sample_data[column]
            transformation = generate_equal_intervals(col_vals, n_bins, column)
            self.transforms[column] = transformation

    def apply_transform(self, data):
        transformed_data = pd.DataFrame()
        for column in data:
            col_vals = data[column]
            transformed = self.transforms[column].transform_data(col_vals)
            transformed_data = pd.concat([transformed_data, transformed], axis=1)

        return transformed_data
            

