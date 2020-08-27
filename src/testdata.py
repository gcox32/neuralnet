import numpy as np

np.random.seed(15)

def create_spiral_data(points, classes):
    """
    create spiral data set 
    params
    ----------
    points (int) : number of points for each class
    classes (int) : number of distinct classes, evenly distributed

    returns
    ---------
    X (numpy array) : array of shape (points*classes, 2)
    y (numpy array) : 1d array of classes
    """
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number + 1))
        r = np.linspace(0.0, 1, points) # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

def create_data(points, classes):
    pass