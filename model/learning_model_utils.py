"""
learning_model_utils provide support functions for learning_model,
learning_problem, and session.
"""
import numpy as np
import pandas as pd
import json

def to_one_shot(labels, num_classes):
    """
    Generates a one shot version of labels.
    """
    num_train = labels.shape[0]
    one_shot = pd.DataFrame(0, index=np.arange(num_train),
                            columns=np.arange(num_classes))
    for i in range(num_train):
        one_shot[labels[i]][i] = 1

    return one_shot

def get_model_path(parent_path, name):
    """
    Returns path to directory allocated for model.
    """
    return parent_path + "/" + name + "/"
    


def record_model(path, model_info):
    """
    Records model into path.
    """
    with open(path, 'w') as file:
        file.write(json.dumps(model_info))


def get_model(path):
    """
    Retrieves model from path.
    """
    with open(path, 'r') as file:
        model_info = json.load(file)
        
    return model_info
