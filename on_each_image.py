from __future__ import division
import struct
import numpy as np
from scipy import misc, stats
import scipy.interpolate as spi
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.feature_extraction import image 
from sklearn import decomposition
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from skimage.transform import resize

# maximum sample to kmeans
SAMP_SIZE_MAX = 1800
# how many kmeans clusters on each image's colors
N_CLUSTERS = 7
# quantiles to get of each color histogram
DEFAULT_QUANTILES = (5, 25, 50, 75, 95)
# the patch size for moving window stats
X_PATCH, Y_PATCH = 16,16
# resize to this x pixels then maintain aspect ratio
X_DOWNSAMPLED = 64


def standardize(image_object):
    """ 
    standardize(image_object, fill_value=255)

    Removes the alpha channel.
    Resizes to X_DOWNSAMPLED as x
    Maintains aspect ratio

    Parameters:
    image_object: 3-d or 2-d array
    """
    if len(image_object.shape) == 3:
        w, h, d = tuple(image_object.shape)
        if d >= 3:
            # Ignore alpha for now
            image_object = image_object[:,:,:3]
            d = 3
    else:
        w, h = image_object.shape
        d = 3
        image_object2 = np.empty((w,h,3))
        for idx in range(d):
            image_object2[:,:,idx] = image_object
        image_object = image_object2
    ylen = int(X_DOWNSAMPLED * h / w + 0.5)
    resized = resize(image_object, (X_DOWNSAMPLED, ylen))
    row_count = np.prod(resized.shape[:2])
    resized_flat = resized.reshape((row_count, d))
    return (image_object, resized, resized_flat)


def histogram(img_flat, percents):
    """ Percentiles of each color column in img_flat"""
    return np.array([np.percentile(img_flat[:,i], percents) for i in range(3)])


def histo_diff(histo1, histo2):
    """ Sum of squared error between two histograms."""
    d = (histo1 - histo2) ** 2.0
    return d.sum()


def hash_ordered_ints(one_zero, chunk_size=256):
    chunks = []
    for idx in range(0, len(one_zero) - chunk_size, chunk_size):
        
        num = int(one_zero[idx+chunk_size - 1])
        for poww in range(0, chunk_size):
            if one_zero[idx + poww]:
                num += 2 ** poww
        chunks.append(hex(num))
    return chunks


def luminosity_grayscale(img_flat):
    """ luminosity_grayscale of img_flat
    as in http://en.wikipedia.org/wiki/Luma_%28video%29
    """
    return img_flat[:,0] * .21 + .72 * img_flat[:,1] + 0.07 * img_flat[:,2]


def perceptive_hash(img_flat):
    """perceptive_hash of img_flat 

    Parameters:
        img_flat: N X 3 array
    """
    img_flat2 = luminosity_grayscale(img_flat)
    mn = img_flat2.mean()
    img_flat2[img_flat2 < mn] = 0
    img_flat2[img_flat2 >= mn] = 1
    return hash_ordered_ints(img_flat2)


def by_patch_stats(img, histo, window, patches=None):
    """
    by_patch_stats(img, window, patches=None, histo=None)

    Find the colors that are most representative
    and most anomalous in the image by searching windows and
    comparing histograms. 

    Parameters:
        img: N X 3 array 
        histo: array
            histogram of full image
        window: tuple
        patches: list 
            (if already computed)
        
    """
    if patches is None:
        patches = image.extract_patches_2d(img, window)
    com_sse = 1e15
    odd_sse = 0
    for idx in range(patches.shape[0]):
        pat = patches[idx].reshape(window[0] * window[1], img.shape[2])
        local_histo = histogram(pat, DEFAULT_QUANTILES)
        diff = histo_diff(histo, local_histo)
        if diff > odd_sse:
            odd_sse = diff 
            odd = local_histo
        if diff < com_sse:
            com_sse = diff
            com = local_histo
    ret = { 
            'com_sse': com_sse,
            'odd_sse': odd_sse,
        }
    for idx in range(3):
        ret['common_histo_%s'%idx] = com[idx] 
        ret['odd_histo_%s'%idx] = odd[idx]             
    return (ret, patches)


def on_each_image(image_object=None, img_name=None):
    """on_each_image(image_object=None, img_name=None)

     Outputs a dictionary of numpy arrays 
    for each image_object or img_name (file).

    Parameters:
        image_object: 2-D or 3-D array 
            image matrix
        img_name: str 
            image filename

        Supply one of the above"""
    if image_object is None:
        # do we assume this can read every
        # image, or fallback....
        image_object = misc.imread(img_name)
    _, image_smaller, img_flat = standardize(image_object)
    w, h, d = image_smaller.shape
    full_histo = histogram(img_flat, DEFAULT_QUANTILES)
    km = KMeans(n_clusters=N_CLUSTERS, 
                    random_state=0)
    image_array_sample = shuffle(img_flat, random_state=0)
    km.fit(image_array_sample[:SAMP_SIZE_MAX])
    # The following slows it down a lot....
    #patch_window =  (int(w / X_PATCH), int(h  / Y_PATCH))
    #ret, patches = by_patch_stats(image_smaller, 
     #                           full_histo,
      #                           patch_window, 
       #                          patches=None)
    pca = decomposition.PCA(copy=True, n_components=None, whiten=False) 
    color_pca = pca.fit(img_flat)
    phash = perceptive_hash(img_flat)
    cens = {'cen_%s'%i: km.cluster_centers_[i] for i in range(N_CLUSTERS)}
    ret = {}
    ret.update({
        'pca_var': color_pca.explained_variance_,
        'pca_fac': pca.components_,
        'histo': full_histo,
        'phash': phash,
    })
    ret.update(cens)
    return ret


def test_data():
    import os
    import datetime
    global r
    tm  = './test_images'
    r = []
    s = datetime.datetime.now()
    print(s)
    for idx, f in enumerate(os.listdir(tm)):
        
        print(idx,f, datetime.datetime.now() - s)
        s = datetime.datetime.now()
        r.append(on_each_image(img_name=os.path.join(tm,f)))
    return r
if __name__ == "__main__":
    r = test_data()
