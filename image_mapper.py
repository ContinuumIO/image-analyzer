from __future__ import division, print_function
from functools import partial
from PIL import Image
from pyspark import SparkConf
from pyspark import SparkContext 
from StringIO import StringIO
import yaml
import numpy as np
import os
import sys
from on_each_image import on_each_image
import search
from load_data import download_zipped_faces
from hdfs_paths import hdfs_path, hdfs_template


# load yaml config from this dir
config_path = os.path.join(os.path.dirname(__file__),'config.yaml')
config = yaml.load(open(config_path))

# set up Spark
conf = SparkConf()
conf.set('spark.executor.instances', 10)
sc = SparkContext('yarn-client', 'pyspark-demo', conf=conf)

# some face images for testing
TEST_DATA = 'http://vasc.ri.cmu.edu/idb/images/face/frontal_images/images.tar'
# keys output in each dictionary for on_each_image.  The values are np.arrays
RESULT_KEYS = ['cen',
              'histo',
             'pca_fac',
             'pca_var',
             'phash']
# Do addFile so remote workers have python code
sc.addFile(os.path.join(os.path.dirname(__file__),'hdfs_paths.py'))
sc.addFile(os.path.join(os.path.dirname(__file__),'on_each_image.py'))
sc.addFile(config_path)
sc.addFile(os.path.join(os.path.dirname(__file__),'search.py'))

# These are options to the flat_map_indicators function
# which can do these mappings.
options_template  = {
        'phash_to_key': False,
        'key_to_phash': False,
        'cluster_to_phash': False,
        'phash_to_cluster': False,
        'flattened_to_phash': False,
        'phash_to_flattened': False,
        'cluster_to_key': False,
        'key_to_cluster': False,
        'flattened_to_key': False,
        'ward_to_cluster': False,
    }
    

def load_image(image):
    """Load one image, where image = (key, blob)"""
    from StringIO import StringIO
    from PIL import Image
    img = Image.open(StringIO(image[1]))
    image_object = np.asarray(img, dtype=np.uint8)
    return  image_object.astype(np.int) 
    

def _map_on_each_image(image):
    """on_each_image with id given in metadata.
    Creates a dictionary with RESULT_KEYS for each image."""
    return on_each_image(config, 
                image_object=load_image(image), 
                phash_offset=0, 
                metadata={'id':image[0]})


def map_on_each_image(input_spec, output_path):
    """Applies on_each_image to each function in input_spec, 
    typically a wildcard hdfs file pattern."""
    img_out = sc.binaryFiles(input_spec).map(_map_on_each_image)
    img_out.saveAsPickleFile(output_path)
    return img_out


def phash_chunks(phash_chunk_len, phashes):
    """Tokenization of perceptive hashes to help
    with searchability on partial images. 
    """
    return [tuple(phashes[idx - phash_chunk_len:idx]) for idx in range(phash_chunk_len, len(phashes))]



def flat_map_indicators(phash_chunk_len,
                        kPoints,
                        options,
                        k, 
                        flattened, 
                        phashes,
                        wards):
    """Returns a list of key value pairs according
    to options.  See options_template above.  
     """
    ph = phash_chunks(phash_chunk_len, phashes)
    items = []
    best_cluster = closestPoint(flattened, kPoints)
    if options.get('phash_to_key'):
        items += [(phi, k) for phi in ph]
    if options.get('key_to_phash'):
        items += [( k, phi) for phi in ph]
    if options.get('phash_to_cluster'):
        items += [(phi, best_cluster) for phi in ph]
    if options.get('cluster_to_phash'):
        items += [(best_cluster, phi) for phi in ph]
    if options.get('phash_to_flattened'):
        items += [(phi, flattened) for phi in ph]
    if options.get('flattened_to_phash'):
        items += [(flattened, phi) for phi in ph]
    if options.get('flattened_to_key'):
        items += [(flattened, k)]
    if options.get('cluster_to_key'):
        items += [(best_cluster, k)]
    if options.get('key_to_cluster'):
        items += [(k, best_cluster)]
    if options.get('ward_to_cluster'):
        items += [(wa, best_cluster) for wa in wards]
    if options.get('ward_to_key'):
        items ++ [(wa, k) for wa in wards]
    return items


def closestPoint(p, centers):
    """Index of closest center in centers to point p """
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


def flatten_hist_cen(x):
    """Flatttens histograms, centroids of individual images, 
    and pca factors of original images to a row vector."""
    return np.concatenate((x['cen'].flatten(),
                            x['histo'].flatten()))


def trim_counts_dict(max_len, data, new_data):
    """Reducers can use this to keep a running counts dictionary
    where the number of keys in memory does not exceed max_len.
    """
    d2 = {}
    for k in set(new_data.keys() + data.keys()):
        d2[k] = new_data.get(k, 0) + data.get(k, 0)
    data = d2
    if len(data) > max_len:
        for k,v in sorted(data.items(), key=lambda x:x[1]):
            data.pop(k)
            if len(data) < max_len:
                break
    return data


def km_map(kPoints, p):
    """For point p, find closest cluster idx.
    Emit a count dictionary of perceptive hashes"""
    closest_idx = closestPoint(p[1], kPoints)
    phash_counter= {phash: p[2].count(phash) for phash in p[2]}
    ward_counter= {wa: p[3].count(phash) for wa in p[3]}
    return (closest_idx, (p[1], 1, phash_counter, ward_counter))


def reduce_dist(ward_max_len, phash_max_len, a, b):
    """Reduce by calculating new points in kmeans and also
    merging the perceptive hash counts dictionary."""
    (x1, y1, z1, wa1) = a
    (x2, y2, z2, wa2) = b
    phashes_union = trim_counts_dict(phash_max_len, z1, z2)
    ward_union = trim_counts_dict(ward_max_len, wa1, wa2)
    return (x1 + x2, y1 + y2, phashes_union, ward_union)


def kmeans(config):
    """ Kmeans with merging and counting of perceptive hashes among
    clusters."""
    measures = sc.pickleFile(hdfs_path(config, 'on_each_image', 'measures'))
    data = measures.map(lambda x:(x[1]['id'], flatten_hist_cen(x[1]), x[1]['phash'], x[1]['ward'])).cache()
    K = config['n_clusters_group']
    convergeDist = config['kmeans_group_converge']
    sample = data.takeSample(False, K, 1)
    kPoints = [k[1] for k in sample]
    tempDist = 10 * convergeDist
    idx = 0
    within_set_sse = []
    while tempDist > convergeDist:
        max_len = config['in_memory_set_len']  / K
        ward_max_len = int(.1 * max_len)
        phash_max_len = int(max_len - ward_max_len)
        closest = data.map(partial(km_map, kPoints))
        pointStats = closest.reduceByKey(partial(reduce_dist, 
                                                ward_max_len, 
                                                phash_max_len))
        pts_hash_union = pointStats.map(
                            lambda (x, (y, z, u, w)): (x, (y / z, u, w)
                        )).collect()
        tempDist = sum(np.sum((kPoints[x] - y) ** 2) for (x, (y, u, w)) in pts_hash_union)
        phash_unions = tuple(u for (x, (y, u, w)) in pts_hash_union)
        ward_unions = tuple(w for (x, (y, u, w)) in pts_hash_union)
        newPoints = tuple((x, np.array(y, dtype="int32")) for (x, (y, u, w)) in pts_hash_union)
        idx += 1
        if idx > config['max_iter_group']:
            break
        print('kmeans did iteration: ', idx, file=sys.stderr)
    for (x, y) in newPoints:
        kPoints[x] = y

    # The rest of the function deals with writing various lookup tables.

    # save the fit data and the meta stats as a single item in list
    kpsave = sc.parallelize([{'kPoints': kPoints,
                            'ss': tempDist,
                            'iter': idx,
                            'within_set_ss': within_set_sse,
                            'phash_unions': phash_unions,
                            'ward_unions': ward_unions,
                            }])
    kpsave.saveAsPickleFile(hdfs_path(config, 'km','cluster_center_meta'))
    
    # do several key value maps for lookups later
    # copy of options_template for what type of flat map
    options = options_template.copy()
    
    # partial func considering the perceptive hash tokenization
    # and kmeans centroids
    flat_map_temp = partial(flat_map_indicators, 
                        config['phash_chunk_len'], 
                        kPoints)
    
    # partial func taking into account what type of flatMap
    options['phash_to_key'] = True
    flat_map = partial(flat_map_temp, options)
    ph_to_k = data.flatMap(lambda x:flat_map(*x))
    ph_to_k.groupByKey()
    
    # saving perceptive hash to image key
    ph_to_k.saveAsPickleFile(hdfs_path(config, 'km', 'phash_to_key'))
    
    # Now save the cluster idx to image key lookup
    options['phash_to_key'] = False
    options['cluster_to_key'] = True
    flat_map = partial(flat_map_temp, options)
    cluster_to_k = data.flatMap(lambda x:flat_map(*x))
    cluster_to_k.groupByKey()
    cluster_to_k.saveAsPickleFile(hdfs_path(config, 'km', 'cluster_to_key'))
    
    # Image key to cluster map
    options['key_to_cluster'] = True
    flat_map = partial(flat_map_temp, options)
    key_to_cluster = data.flatMap(lambda x:flat_map(*x))
    key_to_cluster.saveAsPickleFile(hdfs_path(config, 'km', 'key_to_cluster'))
    
    # Image perceptive hash to cluster idx map
    options['key_to_cluster'] = False
    options['phash_to_cluster'] = True
    flat_map = partial(flat_map_temp, options)
    phash_to_cluster = data.flatMap(lambda x:flat_map(*x))
    phash_to_cluster.groupByKey()
    phash_to_cluster.saveAsPickleFile(hdfs_path(config, 'km', 'phash_to_cluster'))
    
    # Image cluster to perceptive hash chunks flat map
    options['phash_to_key'] = options['phash_to_flattened'] = False
    options['phash_to_cluster'] = False
    options['cluster_to_phash'] = True
    flat_map = partial(flat_map_temp, options)
    cluster_phash = data.flatMap(lambda x:flat_map(*x))
    cluster_phash.groupByKey()
    cluster_phash.saveAsPickleFile(hdfs_path(config, 'km', 'cluster_to_phash'))
    
    # ward cluster to kmeans cluster            
    options['cluster_to_phash'] = False
    options['ward_to_cluster'] = True
    flat_map = partial(flat_map_temp, options)
    cluster_phash = data.flatMap(lambda x:flat_map(*x))
    cluster_phash.groupByKey()
    cluster_phash.saveAsPickleFile(hdfs_path(config, 'km', 'ward_to_cluster'))
    
    # ward  to image key map
    options['ward_to_cluster'] = False
    options['ward_to_key'] = True
    flat_map = partial(flat_map_temp, options)
    cluster_phash = data.flatMap(lambda x:flat_map(*x))
    cluster_phash.groupByKey()
    cluster_phash.saveAsPickleFile(hdfs_path(config, 'km', 'ward_to_cluster'))
        

if __name__ == "__main__":
    actions = config['actions']
    config['candidate_measures_spec'] = hdfs_path(config, 
                                                'candidates', 
                                                'measures' + config['candidate_batch'] 
                                                )
    make_hdfs_dirs(config)
    if 'download' in actions:
        download_zipped_faces(config)
    if  'on_each_image' in actions:
        map_on_each_image(config['input_spec'], 
                          hdfs_path(config, 'on_each_image', 'measures'))
    if  'kmeans' in actions:
        kmeans(config)
    if 'find_similar' in actions:
        search.find_similar(sc, config)
