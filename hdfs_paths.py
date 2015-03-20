from __future__ import division, print_function
import subprocess as sp
import os
hdfs_template = lambda a: 'hdfs://' + os.path.join(*a)

def hdfs_path(*args):
    config = args[0]
    if args[1].startswith('/'):
        p = []
    else:
        p = [os.path.join('/', config['test_name']) ]
    p.extend(args[1:])
    p[-1] = os.path.basename(p[-1])
    return  hdfs_template(p)
    
    
def make_hdfs_dirs(config):
    root = os.path.join('/',config['test_name'])
    paths = [root,
             '%s/map_each_image' % root,
             '%s/km' % root,
             '%s/candidates' % root,
             '%s/candidates/%s' % (root, config['candidate_batch']),
             ]
    if config.get('fuzzy_example_data', False):
        paths.append(config['fuzzy_example_data'])
    for p in paths:
        print(sp.Popen(['hadoop',
                        'fs',
                        '-mkdir',
                        '-p',
                         p,]).communicate())
