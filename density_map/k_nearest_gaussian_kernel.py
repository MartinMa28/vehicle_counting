import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage.filters import gaussian_filter


# For detailed explanation of the density map, please refer to the 2.2 section of
# (https://www.semanticscholar.org/paper/Single-Image-Crowd-Counting-via-Multi-Column-Neural-Zhang-Zhou/427d6d9bc05b07c85fc6b2e52f12132f79a28f6c)

def density_map_generator(img_shape, points, beta=0.3, k=3):
    """
    The function is used to generate a density map for crowds or vehicles.

    args:
        img_size: An iterable object that indicates the shape (height, width) of the density map.
        points: A list of coordinates (row, col) for the targets, which could be the heads of 
            pedestrians or the center of vehicles. e.g. [(104, 44), (22, 87)]
        beta: The coefficient that makes the standard deviation sigma proportional to
            the mean distances of k nearest points.
        k: It represents the "k" of k-nearest neighbors. The number of target points should be greater than k.


    return:
        density: the density map
    """
    density = np.zeros(img_shape, dtype=np.float32)
    point_num = len(points)
    if point_num == 0:
        return density

    # KDTree is used to query the k nearest points for each target.
    # Note that KDTree.query gets the k nearest including the original point.
    tree = KDTree(points.copy(), leafsize=2048)
    dists, indices = tree.query(points, k=k+1)

    for i, pt in enumerate(points):
        single_kernel = np.zeros(img_shape, dtype=np.float32)
        assert pt[0] >= 0 and pt[0] < img_shape[0]
        assert pt[1] >= 0 and pt[1] < img_shape[1]

        single_kernel[int(pt[0]), int(pt[1])] = 1.

        sigma = 1
        if point_num > 1:
            # when there are more than one points, compute the mean distance
            # except to the point itself
            sigma = (dists[i][1:].sum() / k) * beta

        density += gaussian_filter(single_kernel, sigma)
    
    return density