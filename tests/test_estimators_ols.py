import pandas as pd
import numpy as np
import pytest
from deepecon.estimators import OLS


def simu_data() -> pd.DataFrame:
    """Generate a simulated dataset with age column and three x columns (x1, x2, x3),
    where x3 contains zeros and negative values"""
    np.random.seed(42)
    size = (100, 1)

    weight = np.random.normal(65, 12, size=size)
    random_data_1 = np.random.normal(0, 1, size=size)
    height = weight * 0.4 + 140 + random_data_1
    random_data_10 = np.random.normal(0, 10, size=size)
    _simu = np.hstack((height, weight, random_data_1, random_data_10))
    df: pd.DataFrame = pd.DataFrame(_simu, columns=['height', 'weight', 'random_data_1', 'random_data_10'])
    return df


