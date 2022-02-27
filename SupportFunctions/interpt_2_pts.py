import numpy as np


def interpt_2_pts(x: float, x_vec: np.array, y_vec: np.array):
    return(y_vec[0] + (x-x_vec[0])*(y_vec[1]-y_vec[0])/(x_vec[1]-x_vec[0]))