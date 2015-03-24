from __future__ import division, print_function
import struct
from functools import partial
from PIL import Image
import numpy as np
import copy
from StringIO import StringIO
from scipy import misc, stats
from sklearn.cluster import KMeans
from sklearn.feature_extraction.image import extract_patches_2d
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
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering


def standardize(image_object, x_down):
    """ 
    standardize(image_object, x_down)

    Removes the alpha channel.
    Resizes to config['x_down'] as x
    Maintains aspect ratio

    Parameters:
    image_object: 3-d or 2-d array
    """
    w, h, d = tuple(image_object.shape)
    if d == 4:
        
    
        # Ignore alpha for now
        image_object = image_object[:,:,:3]
        d = 3
    elif d == 1:
        w, h = image_object.shape
        d = 3
        # TODO...not sure how to handle
        # 1 band photo
        image_object2 = np.empty((w,h,3))
        image_object2[:,:, 0] = image_object2[:,:, 1] = image_object2[:,:, 2] = image_object / 3.0
        image_object = image_object2
    ylen = int(x_down * h / w + 0.5)
    resized = resize(image_object, (x_down, ylen))
    row_count = np.prod(resized.shape[:2])
    resized_flat = resized.reshape((row_count, d))
    return (image_object, resized, resized_flat)


def phash_chunks(phash_chunk_len, phashes):
    """Tokenization of perceptive hashes to help
    with searchability on partial images. 
    """
    return [tuple(phashes[idx - phash_chunk_len:idx]) for idx in range(phash_chunk_len, len(phashes))]


def flatten_hist_cen(x):
    """Flatttens histograms, centroids of individual images, 
    and pca factors of original images to a row vector."""
    return np.concatenate((x['cen'].flatten(),
                            x['histo'].flatten()))


def histogram(img_flat, percents):
    """ Percentiles of each color column in img_flat"""
    return np.array([np.percentile(img_flat[:,i], percents) for i in range(3)], dtype=np.float32).flatten()


def hash_ordered_ints(bits, one_zero):
    chunks = []
    chunk_size = bits
    for idx in range(0, len(one_zero) - chunk_size, chunk_size):
        chunks.append(hash(tuple(one_zero[idx:idx + chunk_size])))
    return chunks


def luminosity_grayscale(img_flat):
    """ luminosity_grayscale of img_flat
    as in http://en.wikipedia.org/wiki/Luma_%28video%29
    """
    return img_flat[:,0] * .21 + .72 * img_flat[:,1] + 0.07 * img_flat[:,2]


def perceptive_hash(config, img_flat):
    """perceptive_hash of img_flat 

    Parameters:
        img_flat: N X 3 array
    """
    img_flat2 = luminosity_grayscale(img_flat)
    mn = img_flat2.mean()
    img_flat2[img_flat2 < mn] = 0
    img_flat2[img_flat2 >= mn] = 1
    return hash_ordered_ints(config['phash_bits'],img_flat2)


def ward_clustering(config, img_flat):
    X = np.reshape(img_flat, (-1, 1))
    connectivity = grid_to_graph(*img_flat.shape)
    ward = AgglomerativeClustering(
                n_clusters=config['ward_clusters'],
                linkage='ward',
                compute_full_tree = False, 
                connectivity=connectivity).fit(X)
    ulab = np.unique(ward.labels_)
    out = []
    for u in ulab:
        inds = np.where(ward.labels_ == u)[0]
        hsh = hash(tuple(inds - inds[0]))
        out.append(hsh)
    return tuple(out)


def on_each_image_selection(config,
                 image_object=None, 
                  img_name=None, 
                  metadata=None):
    """on_each_image(image_object=None, 
                  img_name=None, 
                  metadata=None)

     Outputs a dictionary of numpy arrays 
    for each image_object or img_name (file).

    Parameters:
        image_object: 2-D or 3-D array 
            image matrix
        img_name: str 
            image filename
        metadata: None or dict 
            other identifiers that need to be in output dict.
        Supply one of the above"""
    if image_object is None:
        image_object = misc.imread(img_file)
    _, image_smaller, img_flat = standardize(image_object, config['x_down'])
    w, h, d = image_smaller.shape
    full_histo = histogram(img_flat, config['quantiles'])
    km = KMeans(n_clusters=config['n_clusters'], 
                    random_state=0)
    image_array_sample = shuffle(img_flat, random_state=0)
    km.fit(image_array_sample[:config['kmeans_sample']])
    pca = decomposition.PCA(copy=True, n_components=None, whiten=True) 
    color_pca = pca.fit(img_flat)
    phash = perceptive_hash(config, img_flat)
    ret = {}
    _, ward_smaller, smaller_flat = standardize(image_smaller, config['ward_x_down'])
    ret.update({
        'pca_var': color_pca.explained_variance_.flatten(),
        'pca_fac': pca.components_.flatten(),
        'histo': full_histo,
        'phash': phash,
        'ward': ward_clustering(config, smaller_flat),
        'cen': np.array(km.cluster_centers_, dtype=np.float32).flatten(),
    })
    if metadata:
        ret.update(metadata)
    if img_name:
        ret['img_name'] = img_name
    return ret


def standardize_image_meta(img):
    m2 = {}
    for hi_key, obj in ((i, getattr(img, i, {}).items()) for i in ('info', 'app')):
        for k, v in obj :
            m2['%s_%s' %(hi_key, k) ] = v
    m2['format'] = getattr(img, 'format', '')
    m2['format_description'] = getattr(img, 'format_description', '')
    return m2

def load_image(config, image):
    """Load one image, where image = (key, blob)"""
    from StringIO import StringIO
    from PIL import Image
    img = Image.open(StringIO(image[1]))
    img_patches = []
    if config.get('patch'):
        window = [int(wf * sz) for wf,sz in zip(img.size, config['patch']['window_as_fraction'])]
        img_patches=extract_patches_2d(
            np.asarray(img), 
            window, 
            max_patches=config['patch']['max_patches'], 
            random_state=config['random_state'])
    image_object = np.asarray(img, dtype=np.uint8)
    meta = standardize_image_meta(img)
    return  image_object, img_patches, meta
    

def on_each_image(config, image):
    """on_each_image with id given in metadata.
    Creates a dictionary with RESULT_KEYS for each image.

    config: dict
        Typically from config.yaml
    image: tuple
        Filename, blob to load by PIL 
        """
    img, quads, meta = load_image(config,image)
    meta2 = copy.deepcopy(meta)
    meta2['is_full'] = False
    meta['is_full'] = True
    out = []
    for met, img_to_measure in [(meta, img)] + [(meta2, q) for q in quads]:
        on_im = on_each_image_selection(config, 
                    image_object=img_to_measure, 
                    metadata={'id':image[0]})
        on_im['meta'] = met 
        out.append((on_im['id'], on_im))
    return out


def map_each_image(sc, config, input_spec, output_path):
    """Applies on_each_image to each function in input_spec, 
    typically a wildcard hdfs file pattern.

    Parameters:
    sc : SparkContext
    config: dictionary
        Typically from config.yaml
    input_spec: str 
        An hdfs wildcard pattern as input images.
    output_path:
        An hdfs dir into which to put the results
    """
    img_out = sc.binaryFiles(
                input_spec
            ).flatMap(
                partial(on_each_image, config)
                )
    img_out.saveAsPickleFile(output_path)
    return img_out

def example(filename,config=None):
    """ For local testing of the image measurements."""
    from StringIO import StringIO
    s = StringIO()
    s.write(open(filename).read())
    image_object = load_image({},(filename,s.getvalue()))
    if config is None:
        config = {
                'ward_x_down': 64, 
                'x_down': 256,
                'quantiles':(50,),
                'n_clusters': 5,
                'ward_clusters': 5,
                'kmeans_sample': 2000,
                'phash_bits': 256
            }
    output = on_each_image_selection(config,
            image_object=image_object[0], 
            metadata={'id': filename})
    return image_object, output