#import torch
import numpy as np
import sklearn.mixture
import scipy.stats
import cv2
from . import bmm


def threshold(array, tau):
    """
    Threshold an array using either hard thresholding, Otsu thresholding or beta-fitting.

    If the threshold value is fixed, this function returns
    the mask and the threshold used to obtain the mask.
    When using tau=-1, the threshold is obtained as described in the Otsu method.
    When using tau=-2, it also returns the fitted 2-beta Mixture Model.


    :param array: Array to threshold.
    :param tau: (float) Threshold to use.
                Values above tau become 1, and values below tau become 0.
                If -1, use Otsu thresholding.
		If -2, fit a mixture of 2 beta distributions, and use
		the average of the two means.
    :return: The tuple (mask, threshold).
             If tau==-2, returns the tuple (mask, otsu_tau, ((rv1, rv2), (pi1, pi2))).
             
    """
    if tau == -1:
        # Otsu thresholding
        minn, maxx = array.min(), array.max()
        array_scaled = ((array - minn)/(maxx - minn)*255) \
            .round().astype(np.uint8).squeeze()
        tau, mask = cv2.threshold(array_scaled,
                                  0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tau = minn + (tau/255)*(maxx - minn)
        # print(f'Otsu selected tau={tau_otsu}')
    elif tau == -2:
        array_flat = array.flatten()
        ((a1, b1), (a2, b2)), (pi1, pi2), niter = bmm.estimate(array_flat, list(range(2)))
        rv1 = scipy.stats.beta(a1, b1)
        rv2 = scipy.stats.beta(a2, b2)
        
        tau = rv2.mean()
        mask = cv2.inRange(array, tau, 1)

        return mask, tau, ((rv1, pi1), (rv2, pi2))
    else:
        # Thresholding with a fixed threshold tau
        mask = cv2.inRange(array, tau, 1)

    return mask, tau


def cluster(array, n_clusters, max_mask_pts=np.infty):
    """
    Cluster a 2-D binary array.
    Applies a Gaussian Mixture Model on the positive elements of the array,
    and returns the number of clusters.
    
    :param array: Binary array.
    :param n_clusters: Number of clusters (Gaussians) to fit,
    :param max_mask_pts: Randomly subsample "max_pts" points
                         from the array before fitting.
    :return: Centroids in the input array.
    """

    array = np.array(array)
    
    assert array.ndim == 2

    coord = np.where(array > 0)
    y = coord[0].reshape((-1, 1))
    x = coord[1].reshape((-1, 1))
    c = np.concatenate((y, x), axis=1)
    if len(c) == 0:
        centroids = np.array([])
    else:
        # Subsample our points randomly so it is faster
        if max_mask_pts != np.infty:
            n_pts = min(len(c), max_mask_pts)
            np.random.shuffle(c)
            c = c[:n_pts]

        # If the estimation is horrible, we cannot fit a GMM if n_components > n_samples
        # covariance type : diagonal, spherical, tied and full
        n_components = max(min(n_clusters, x.size), 1)
        centroids = sklearn.mixture.GaussianMixture(n_components=n_components, n_init=1, covariance_type='full').fit(c).means_.astype(np.int)
        # use Variational Bayes
        #centroids = sklearn.mixture.BayesianGaussianMixture(n_components=n_components, random_state=42).fit(c).means_.astype(np.int)

    return centroids



def paint_circles(img, points, color='red', crosshair=False):
    """
    Paint points as circles on top of an image.

    :param img: BGR image (numpy array).
                Must be between 0 and 255.
                First dimension must be color.
    :param centroids: List of centroids in (y, x) format.
    :param color: String of the color used to paint centroids.
                  Default: 'red'.
    :param crosshair: Paint crosshair instead of circle.
                      Default: False.
    :return: Image with painted circles centered on the points.
             First dimension is be color.
    """

    if color == 'red':
        #color = [255, 0, 0]
        color = [0, 0, 255] # BGR
    elif color == 'white':
        color = [255, 255, 255]
    else:
        raise NotImplementedError(f'color {color} not implemented')

    points = points.round().astype(np.uint16)

    img = np.moveaxis(img, 0, 2).copy()
    if not crosshair:
        for y, x in points:
            img = cv2.circle(img, (x, y), 3, color, -1)
    else:
        for y, x in points:
            img = cv2.drawMarker(img,
                                 (x, y),
                                 color, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
    img = np.moveaxis(img, 2, 0)

    return img



def get_confusion_matrix(label, pred, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()

    ignore_index = label != ignore
    label = label[ignore_index]
    pred = pred[ignore_index]

#    index = (seg_gt * num_class + seg_pred).astype('int32')
#    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    #for i_label in range(num_class):
    #    for i_pred in range(num_class):
    #        #cur_index = i_label * num_class + i_pred
    #        #if cur_index < len(label_count):
    #        if condition:
    #            confusion_matrix[i_label, i_pred] += 1
    #              result = np.zeros((K, K))

    for i in range(len(label)):
      confusion_matrix[label[i]][pred[i]] += 1
      
    return confusion_matrix