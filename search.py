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
    return [(best_cluster , (ch, id_)) for ch in chunks]

    
def count_keys(x):
    """When keys to training images are found to be 
    potentially matching a candidate, this reducer will 
    count the 'votes' for each of training image hashes"""

    counts = {}
    for u in x[1]:
        if u in counts:
            counts[u] +=  1
        else:
            counts[u] = 1
    best = sorted(counts.items(), key=lambda x:-x[1])[0]
    return (x[0], (best, counts))


def join_nearest(sc, 
                config, 
                kPoints, 
                phash_unions, 
                ward_unions, 
                scores):
    
    """Use candidates' scores to assign them to best clusters based 
    on euclidean distance and number of matching hashes, ward or perceptive.
    Join those assigned clusters to perceptive and ward hashes from training
    and then join hashes to keys."""
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
        # TODO I think the following line has a typo distances[b] is what is should be.
        return [(x[0], (b, 'self', distances[best[-1]], x[1]['ward'], x[1]['phash'])) for b in best]
    best_clusters = scores.flatMap(_best_cluster)
    best_clusters
    best_clusters.sortBy(lambda x:x[1][2]).cache()
    phash_c = best_clusters.flatMap(
                    partial(cluster_chunk, 
                                config,
                                'phash'))
    phash_c_id = phash_c.map(lambda x:x[1])
    ward_c = best_clusters.flatMap(partial(cluster_chunk, 
                                        config,
                                        'ward'))
    ward_c_id = ward_c.map(lambda x:x[1])
    cluster_to_phash = sc.pickleFile(hdfs_path(config, 
                                                'km', 
                                                'cluster_to_phash'))
    cluster_to_ward = sc.pickleFile(hdfs_path(config, 
                                            'km', 
                                            'cluster_to_ward'))
    rdds = ( ward_c, phash_c)
    rdds2 = (ward_c_id, phash_c_id)
    table_names = ('ward_matches','phash_matches')
    labels = ('ward_to_key','phash_to_key')
    out = {}
    to_join = []
    for table, rdd, rdd2, label in zip(table_names, rdds, rdds2, labels):
    
        join_on_cluster = rdd.join(
            cluster_to_phash if table == 'phash_matches' else cluster_to_ward
        )
        map_ward_or_phash = join_on_cluster.map(lambda x:(x[1][0][0], x))
        to_key = sc.pickleFile(hdfs_path(config, 'km', label))
        hash_joined = map_ward_or_phash.join(
            to_key
        )
        hash_joined2 = rdd2.join(to_key)
        
        # pulling the two image keys out into pairs
        cand_key_to_key = hash_joined.map(
            lambda x: (x[1][0][1][0][1], x[1][-1])
        )
        samp = cand_key_to_key.take(config['search_sample_step'])
        out[table] = samp
        as_key_counts = cand_key_to_key.groupByKey(
            ).map(
            count_keys
            )
        as_key_counts.cache()
        as_key_counts.saveAsPickleFile(
            hdfs_path(config, 'candidates', config['candidate_batch'], "%s_counts" % label)
        )
        to_join.append(as_key_counts)
    # map the candidate id with best match of a hash with indicators of fit
    def map_best(x):
        """The key, (best agreeing key, vote count for agreeing, total votes) """
        (key, ((best_match, agree_count), dict_)) = x
        return (key, (best_match, agree_count, sum(dict_.values())))
    
    # join the ward best key with phash best key
    joined_final_matches = to_join[0].map(
                            map_best
                        ).join(
                            to_join[1].map(map_best)
                        )
    joined_final_matches.saveAsPickleFile(
            hdfs_path(config, 'candidates',config['candidate_batch'], 'joined_final_matches')
        )
    out['joined'] = joined_final_matches.take(config['search_sample_step'])
    return out


def find_similar(sc, config):
    """Use cluster to hash and hash to key 
    joins to find_similar images.

    TODO: more rounds of search, and have an 
    option to do ward OR perceptive hash OR both.
    Ward is more expansive (false positives) than 
    perceptive hashes, so the join can get slow with many
    matches.  Maybe ward hashes should be a second try."""
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
    scores.cache()
    for net_round in range(config['search_rounds']):
        samples = join_nearest(sc,
                                config,
                                kPoints, 
                                phash_unions,
                                ward_unions,
                            scores)
        
        #TODO logic here for more rounds of sampling
    return samples
