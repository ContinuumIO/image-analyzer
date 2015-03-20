from __future__ import division, print_function
import numpy as np
from hdfs_paths import hdfs_path
from map_each_image import map_each_image, flatten_hist_cen
from functools import partial
from map_each_image import phash_chunks


def cluster_chunk(config, ward_or_phash, x): 
    """Flatten phash or ward clusters """
    (id_, (best_cluster, self, distance, ward, phash)) = x
    if ward_or_phash == 'ward':
        chunks = ward
    else:
        chunks = phash_chunks(config['phash_chunk_len'], phash)
    return [(best_cluster , (ch, id_, ward_or_phash)) for ch in chunks]
    

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
        return [(x[0], (b, 'self', distances[best[-1]], x[1]['ward'], x[1]['phash'])) for b in best]
    best_clusters = scores.flatMap(_best_cluster)
    best_clusters.cache()
    best_clusters.sortBy(lambda x:x[1][2])
    phash_c = best_clusters.flatMap(
                    partial(cluster_chunk, 
                                config,
                                'phash'))
    ward_c = best_clusters.flatMap(partial(cluster_chunk, 
                                        config,
                                        'ward'))
    bc2 = best_clusters.map(lambda x: (x[1][0], x))
    cluster_to_phash = sc.pickleFile(hdfs_path(config, 
                                                'km', 
                                                'cluster_to_phash')).join(bc2)
    cluster_to_ward = sc.pickleFile(hdfs_path(config, 
                                            'km', 
                                            'cluster_to_ward')).join(bc2)

    rdds = ( ward_c, phash_c)
    table_names = ('ward_matches','phash_matches')
    labels = ('ward_to_key','phash_to_key')
    out = {}
    for table, rdd, label in zip(table_names, rdds, labels):
    
        rdd.join(
                cluster_to_phash
            .map(
                lambda x: (x[1][1],x)
            ).join(
                sc.pickleFile(hdfs_path(config, 'km', label))
            )
        )
        samp = rdd.take(config['search_sample_step'])
        path = hdfs_path(config, 'candidates',config['candidate_batch'], table)
        out[table] = samp
        rdd.saveAsPickleFile(path)
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
        from pprint import pformat
        print(pformat(samples)[:30000])
        
