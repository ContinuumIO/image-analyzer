from __future__ import division, print_function
import subprocess as sp
hdfs_template = 'hdfs:///%(test_name)s/%(test_part)s%(second_test_part)s/%(fname)s'

def hdfs_path(*args):
    config = args[0]
    if len(args) == 3:
        test_part, fname = args[1:]
        second_test_part = ''
    else:
        test_part,, second_test_part, fname = args[1:]
    h = hdfs_template % {'test_part': test_part,
                        'fname': fname,
                        'second_test_part': second_test_part,
                        'test_name':config['test_name']}
    if not fname:
        return h[:-1]
    return h

    
def make_hdfs_dirs(config):
    paths = ['/' + config['test_name'],
             '/%s/on_each_image' % config['test_name'],
             '/%s/km' % config['test_name'],
             '/%s/candidates' % config['test_name'],
             '/%s/candidates/%s' % config['candidate_batch'],]
    for p in paths:
        print(sp.Popen(['hadoop',
                        'fs',
                        '-mkdir',
                        '-p',
                         p,]).communicate())
