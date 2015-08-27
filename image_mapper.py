from __future__ import division, print_function
from functools import partial
from PIL import Image
from pyspark import SparkConf
from pyspark import SparkContext 
from StringIO import StringIO
import yaml
import numpy as np
import os
import operator 
import sys
from map_each_image import map_each_image, flatten_hist_cen, phash_chunks
import search
from hdfs_paths import hdfs_path, make_hdfs_dirs


# load yaml config from this dir
config_path = os.path.join(os.path.dirname(__file__),'config.yaml')
config = yaml.load(open(config_path))

# set up Spark
conf = SparkConf()
conf.set('spark.executor.instances', 8)
sc = SparkContext('yarn-client', 'pyspark-demo', conf=conf)

# keys output in each dictionary for map_each_image.  The values are np.arrays
RESULT_KEYS = ['cen',
              'histo',
              'ward',
             'pca_fac',
             'pca_var',
             'phash']
# Do addFile so remote workers have python code
sc.addFile(os.path.join(os.path.dirname(__file__),'hdfs_paths.py'))
sc.addFile(os.path.join(os.path.dirname(__file__),'map_each_image.py'))
sc.addFile(config_path)
sc.addFile(os.path.join(os.path.dirname(__file__),'search.py'))
sc.addFile(os.path.join(os.path.dirname(__file__),'fuzzify_training.py'))

# These are options to the flat_map_indicators function
# which can do these mappings.
options_template  = {
        'cluster_to_flattened':True,
        'cluster_to_key': True,
        'cluster_to_phash': True,
        # TODO it would be more efficient to combine
        # cluster_to_phash with cluster_to_ward
        'cluster_to_ward': True,
        'flattened_to_cluster': True,
        'flattened_to_key': True,
        'flattened_to_phash': True,
        'key_to_cluster': True,
        'key_to_phash': True,
        'phash_to_cluster': True,
        'phash_to_flattened': True,
        'phash_to_key': True,
        'ward_to_cluster': True,
        'ward_to_key': True,
    }
    

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
    if options.get('cluster_to_flattened'):
        items += [(best_cluster, flattened)]
    if options.get('key_to_cluster'):
        items += [(k, best_cluster)]
    if options.get('ward_to_cluster'):
        items += [(wa, best_cluster) for wa in wards]
    if options.get('cluster_to_ward'):
        items += [(best_cluster, wa) for wa in wards]
    if options.get('ward_to_key'):
        items += [(wa, k) for wa in wards]
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
    Emit that with a count dictionary of perceptive hashes
    and same for ward hashes"""
    closest_idx = closestPoint(p[1], kPoints)
    phash_counter= {phash: p[2].count(phash) for phash in p[2]}
    ward_counter= {wa: p[3].count(wa) for wa in p[3]}
    return (closest_idx, (p[1], 1, phash_counter, ward_counter))


def reduce_dist(ward_max_len, phash_max_len, a, b):
    """Reduce by calculating new points in kmeans and also
    merging the perceptive hash and ward hash counts dictionaries."""
    (x1, y1, z1, wa1) = a
    (x2, y2, z2, wa2) = b
    phashes_union = trim_counts_dict(phash_max_len, z1, z2)
    ward_union = trim_counts_dict(ward_max_len, wa1, wa2)
    return (x1 + x2, y1 + y2, phashes_union, ward_union)


def kmeans(config):
    """ Kmeans with merging and counting of perceptive hashes and 
    ward hashes among clusters."""
    measures = sc.pickleFile(hdfs_path(config, 'map_each_image', 'measures'))
    data = measures.map(lambda x:(x[1]['id'], flatten_hist_cen(x[1]), x[1]['phash'], x[1]['ward'])).cache()
    K = config['n_clusters_group']
    convergeDist = config['kmeans_group_converge']
    sample = data.takeSample(False, K, 1)
    kPoints = [k[1] for k in sample]
    if convergeDist is not None:
        tempDist = 10 * convergeDist
    else:
        tempDist = 1e12
    idx = 0
    within_set_sse = []
    while tempDist > convergeDist:
        max_len = config['in_memory_set_len']  / K
        ward_max_len = int(.5 * max_len)
        phash_max_len = int(max_len - ward_max_len)
        closest = data.map(partial(km_map, kPoints))
        pointStats = closest.reduceByKey(partial(reduce_dist, 
                                                ward_max_len, 
                                                phash_max_len))
        pts_hash_union = pointStats.map(
                            lambda (x, (y, z, u, w)): (x, (y / z, u, w)
                        ))
        if convergeDist is not None:
            tempDist = pts_hash_union.map(
                lambda (x, (y, u, w)): np.sum((kPoints[x] - y) ** 2)
            ).sum()
        newPoints = pts_hash_union.map(
                lambda (x, (y, u, w)): (x, np.array(y, dtype="int32"))
                ).collect()
        idx += 1
        if idx > config['max_iter_group']:
            break
        print('kmeans did iteration: ', idx, file=sys.stderr)
    for (x, y) in newPoints:
        kPoints[x] = y
    phash_unions = pts_hash_union.map(
                    lambda (x, (y, u, w)): u
                )
    phash_unions.saveAsPickleFile(hdfs_path(config, 'km', 'phash_unions'))
    ward_unions =  pts_hash_union.map(
                lambda (x, (y, u, w)): w
            )
    ward_unions.saveAsPickleFile(hdfs_path(config, 'km', 'ward_unions'))
    # The rest of the function deals with writing various lookup tables.

    # save the fit data and the meta stats as a single item in list
    kpsave = sc.parallelize([kPoints,
                            tempDist,
                            within_set_sse,
                            ])
    kpsave.saveAsPickleFile(hdfs_path(config, 'km','cluster_center_meta'))
    
    def flat(field_to_field):
        flat_map = partial(flat_map_indicators, 
                        config['phash_chunk_len'], 
                        kPoints,
                        {field_to_field:True})
        data.flatMap(
                lambda x: flat_map(*x)
            ).saveAsPickleFile(
                hdfs_path(config, 'km', field_to_field)
            )
    options = options_template.copy()
    options.update(config['kmeans_output'])
    for k, v in options.items():
        if v:
            flat(k)
        

if __name__ == "__main__":
    if config.get('random_state'):
        config['random_state'] = np.random.RandomState(config['random_state'])
    else:
        config['random_state'] = np.random.RandomState(None)
    import datetime
    started = datetime.datetime.now()
    print('started at:::', started)
    actions = config['actions']
    make_hdfs_dirs(config)
    if  'map_each_image' in actions:
        map_each_image(sc, config, config['input_spec'], 
                          hdfs_path(config, 'map_each_image', 'measures'))
    if  'kmeans' in actions:
        kmeans(config)
    if 'find_similar' in actions:
        search.find_similar(sc, config)
    ended = datetime.datetime.now()
    print('Elapsed Time (seconds):::', 
            (ended - started).total_seconds(),
            '\nAt', 
            ended.isoformat())
