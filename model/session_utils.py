import numpy as np
import pandas as pd

def empty_df(epochs, columns):
    return pd.DataFrame(0.0, index=np.arange(epochs), columns=columns)
