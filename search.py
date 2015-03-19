from __future__ import division, print_function
import numpy as np
from hdfs_paths import hdfs_path
from map_each_image import map_each_image
from functools import partial
def _candidate_map_1(x):
    (cluster, (db_phash, candidate_phash)) = x
    return (cluster, len(set(candidate_phash).intersection(set(db_phash))))


def _candidate_map_2(x):
    (cluster, (db_ward, candidate_ward)) = x
    return (cluster, len(set(candidate_ward).intersection(set(db_ward))))


def cluster_to_chunk(ward_or_phash, x): 
    (id_, (data, best_cluster)) = x
    return [((best_cluster, ch ), (_id, ward_or_phash)) for ch in data[ward_or_phash]]


def scores_to_flat(x):
    (id_, (data, best_cluster)) = x
    return ((bc, flatten_hist_cen_pca(data)), (_id, 'flat'))


def dist(kPoints, x): 
    ((cluster, flattened), (_id, typ)) = x
    dist = np.sum((flattened - kPoints[cluster]) ** 2)
    return ((dist, cluster,), ( _id, typ))


def join_nearest(sc, 
                config, 
                kPoints, 
                phash_unions, 
                ward_unions, 
                scores):
    
    def _best_cluster(phash_unions, ward_unions, ward_or_phash, kPoints, x):
        counts =[]
        for unions in (phash_unions, ward_unions):
            counts.append([])
            for u in unions:
                p = 0
                for item in x[1][ward_or_phash]:
                    p += u.get(item, 0)
                counts[-1].append(p)
        best = list(map(np.argmax, counts))
        best.append(np.argmin([kPoints[i] - flatten_hist_cen_pca(x[1]) for i in range(len(kPoints))]))
        return [(x[0], b) for b in best]
    best_clusters = scores.flatMap(_best_cluster)
    flattened = scores.join(best_clusters).flatMap(scores_to_flat)
    flattened.join(
        sc.pickleFile(
            hdfs_path(config, 'km', 'cluster_to_flattened')
        ).map(
            lambda x:x)
    )
    phash_chunks = scores.foreach(
                    partial(cluster_to_chunk, 
                                'phash'))
    ward_chunks = scores.foreach(partial(cluster_to_chunk, 
                                        'ward'))
    cluster_to_phash = sc.pickleFile(hdfs_path(config, 
                                                'km', 
                                                'cluster_to_phash'))
    cluster_to_ward = sc.pickleFile(hdfs_path(config, 
                                            'km', 
                                            'cluster_to_ward'))
    phash_chunks.join(cluster_to_phash.map(lambda x: x))
    ward_chunks.join(cluster_to_ward.map(lambda x: x))
    
    ward_sample = ward_chunks.take(config['search_sample_step']), 
    phash_sample = phash_chunks.take(config['search_sample_step'])
    return {'euclidean': distance_sample,
            'ward': ward_sample,
            'phash': phash_sample,}


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
        print(samples)
        # TODO search more rounds of lookups.

        
