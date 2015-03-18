from __future__ import division, print_function
import numpy as np
def _candidate_map_1(x):
    (cluster, (db_phash, candidate_phash)) = x
    return (cluster, len(set(candidate_phash).intersection(set(db_phash))))


def _candidate_map_2(x):
    (cluster, (db_ward, candidate_ward)) = x
    return (cluster, len(set(candidate_ward).intersection(set(db_ward))))


def candidate_map(sc, config, kPoints, phash_unions, x):
    id_ = x['id']
    keys_to_ward = None
    keys_to_phash = None
    if do_phash:
        phash_counts =[]
        for phu in phash_unions:
            p = 0
            for phash in x['phash']:
                ph += phu.get(phash, 0)
            phash_counts.append(ph)
        pch = phash_chunks(config['phash_chunk_len'], 
                          x['phash'])
        phash_chunks = sc.parallelize([(np.argmax(phash_counts), ph ) for ph in pch ])
        flattened = flatten_hist_cen_pca(x)
        distances = [np.sum((p - kPoints[i]) ** 2) for i in range(len(kPoints))]
        best_idx_km = np.argmin(distances)
        cp_map = partial(cluster_phash_filter, best_idx_km, best_idx_ph)
        cluster_to_phash = sc.pickleFile(hdfs_path('km', 'cluster_to_phash'))
        phash_to_key = sc.pickleFile(hdfs_path('km', 'phash_to_key')).map(lambda x:x)
        km_ph_matches = cluster_to_phash.join(phash_chunks)
        km_ph_matches.cache()
        keys_to_phash = km_ph_matches.map(
                            _candidate_map_1
                        ).sortBy(
                            lambda x: - x[1]
                        ).zipWithIndex(
                        ).filter(
                            lambda x: x[1] < config['search_sample_step']
                        ).map(
                            lambda x:(x[0][1], x[0][0])
                        ).join(   # <<< --- right or left join or...? TODO
                            phash_to_key
                        ).map(
                            lambda x:(id_, (x[1], x[0][0]))
                        )
    if do_ward:
        cluster_to_ward = sc.parallelize(hdfs_path('km', 'cluster_to_ward'))
        ward_chunks = sc.parallelize([(w, id_) for w in x['ward']])
        ward_matches = cluster_to_ward.join(ward_chunks)
        keys_to_ward = ward_matches.map(
                            _candidate_map_2
                        ).sortBy(
                            lambda x: - x[1]
                        ).zipWithIndex(
                        ).filter(
                            lambda x: x[1] < config['search_sample_step']
                        ).map(
                            lambda x:(x[0][1], x[0][0])
                        ).join(
                            phash_to_key
                        ).map(
                            lambda x:(id_, (x[1], x[0][0]))
                        )
    return keys_to_phash, keys_to_ward


def find_similar(sc, config):
    kmeans_meta = sc.pickleFile(hdfs_path('km','cluster_center_meta'))
    kmeans_meta = kmeans_meta.map(lambda x:x).collect()[0]
    scores = map_on_each_image(config['candidate_spec'], config['candidate_measures_spec'])
    for net_round in range(config['search_rounds']):
        on_each_candidate = partial(candidate_map, 
                                    config, 
                                    kPoints, 
                                    phash_unions)
        keys_to_phash, keys_to_ward = scores.flatMap(on_each_candidate)
        for table, rdd in zip(('keys_to_phash','keys_to_ward'),(keys_to_phash, keys_to_ward)):
            if rdd is not None:
                rdd.saveAsPickleFile(hdfs_path('candidates', config['candidate_batch'], table))
        if net_round - 1 >= config['search_rounds']:
            break
        # TODO search more rounds of lookups.

    print(cand_key_ph.collect())





