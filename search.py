from __future__ import division, print_function
import numpy as np
from hdfs_paths import hdfs_path
from map_each_image import map_each_image, flatten_hist_cen
from functools import partial
from map_each_image import phash_chunks


def cluster_chunk(config, ward_or_phash, x): 
    """Flatten phash or ward clusters """
    (id_, (data, best_cluster)) = x
    if ward_or_phash == 'ward':
        chunks = data[ward_or_phash]
    else:
        chunks = phash_chunks(config['phash_chunk_len'], data[ward_or_phash])
    return [((best_cluster,ch ), (id_, ward_or_phash)) for ch in chunks]
    

def join_nearest(sc, 
                config, 
                kPoints, 
                phash_unions, 
                ward_unions, 
                scores):
    pw = tuple(zip(('phash', 'ward'), (phash_unions, ward_unions)))
    def _best_cluster(x):
        counts =[]
        for ward_or_phash, unions in pw:
            counts.append([])
            for u in unions:
                p = 0
                for item in x[1][ward_or_phash]:
                    p += u.get(item, 0)
                counts[-1].append(p)
        best = list(map(np.argmax, counts))
        distances = [np.sum((kPoints[i] - flatten_hist_cen(x[1]))**2) for i in range(len(kPoints))]
        best.append(np.argmin(distances))
        return [(x[0], (b, 'self', distances[best[-1]])) for b in best]
    best_clusters = scores.flatMap(_best_cluster)
    best_clusters.cache()
    best_clusters.sortBy(lambda x:x[1][1][2])
    phash_c = best_clusters.flatMap(
                    partial(cluster_chunk, 
                                config,
                                'phash'))
    ward_c = best_clusters.flatMap(partial(cluster_chunk, 
                                        config,
                                        'ward'))
    bc2 = best_clusters.map(lambda x: (x[1][0], (x[0], x[1][1:])))
    cluster_to_phash = sc.pickleFile(hdfs_path(config, 
                                                'km', 
                                                'cluster_to_phash')).join(bc2)
    cluster_to_ward = sc.pickleFile(hdfs_path(config, 
                                            'km', 
                                            'cluster_to_ward')).join(bc2)
    # (cluster, ((b, 'self'), ward))
    phash_c_join = phash_c.join(cluster_to_phash)
    ward_c_join = ward_c.join(cluster_to_ward)
    phash_c_join.map(
            lambda x:(x[1][1], x)
        ).join(
            sc.pickleFile(hdfs_path(config, 'km', 'phash_to_key'))
        )
    ward_c_join.map(
            lambda x:(x[1][1], x)
        ).join(
            sc.pickleFile(hdfs_path(config, 'km', 'ward_to_key'))
        )
    ward_sample = ward_c_join.take(config['search_sample_step']), 
    phash_sample = phash_c_join.take(config['search_sample_step'])
    rdds = ( score_join.map(lambda x:(x[0],x[1][1])), ward_c_join, phash_c_join)
    out = {}
    for table, rdd in zip(('euclidean','ward','phash'), rdds):
        p = hdfs_path(config, config['candidate_batch'], table)
        out[table] = rdd.take(config['search_sample_step'])
    return out
def find_similar(sc, config):
    
    kmeans_meta = sc.pickleFile(hdfs_path(config, 'km','cluster_center_meta'))
    kmeans_meta = kmeans_meta.map(lambda x:x).collect()
    kPoints, tempDist, within_set_sse = kmeans_meta
    phash_unions = sc.pickleFile(
                    hdfs_path(config, 'km', 'phash_unions')
                ).map(
                    lambda x:x
                ).collect()
    ward_unions = sc.pickleFile(
                    hdfs_path(config, 'km', 'ward_unions')
                ).map(lambda x:x).collect()
    if not config.get('candidate_has_mapped'):
        scores = map_each_image(sc, 
                config, 
                config['candidate_spec'], 
                config['candidate_measures_spec'])
    else:
        scores = sc.pickleFile(config['candidate_measures_spec'])
    for net_round in range(config['search_rounds']):
        
        samples = join_nearest(sc,
                                config,
                                kPoints, 
                                phash_unions,
                                ward_unions,
                            scores)
        from pprint import pprint 
        pprint(samples)        
        
