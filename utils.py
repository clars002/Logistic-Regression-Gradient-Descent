import numpy as np


def interpolate(start: int, end: int, num_pts: int):
    diff = end - start
    if diff == 0:
        return np.zeros(num_pts)
        
    step = diff / (num_pts)
    return np.arange(start, end, step)


def make_color_gradient(start_color, stop_color, num_pts):
    color_gradient = np.array([interpolate(start_color[0], stop_color[0], num_pts), interpolate(start_color[1], stop_color[1], num_pts), interpolate(start_color[2], stop_color[2], num_pts)])
    colors = [0] * num_pts
    for i in range(num_pts):
        colors[i] = (float(color_gradient[0][i] / 255), float(color_gradient[1][i] / 255), float(color_gradient[2][i] / 255))
    
    return colors