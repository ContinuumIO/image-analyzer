from __future__ import division, print_function
""" 
First in bash on local machine, do one
step to make temp dir permission for hdfs:

CLUSTER=image_cluster13
rmt (){
	conda cluster run.cmd $CLUSTER "$1"
}
rmt "rm -rf  /tmp/hdfs_tmp && mkdir -p  /tmp/hdfs_tmp && chown -R hdfs /tmp/hdfs_tmp"

Then: submit load_data.py

"""
import requests
import subprocess as sp
from StringIO import StringIO
from fuzzify_training import fuzzify
import os
utmp = "/tmp/anaconda-cluster/hdfs_tmp/"

TEST_DATA = 'http://vasc.ri.cmu.edu/idb/images/face/frontal_images/images.tar'
images_test = os.path.join(utmp, 'images_test')
def download_zipped_faces(config, url=TEST_DATA, fname=images_test):
    from tarfile import TarFile
    
    os.chdir(utmp)
    req = requests.get(url, stream=True)
    f = open(fname, 'w')
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            f.write(chunk)
            f.flush()
    with TarFile.open(fname) as tf:
    	tf.extractall()

	sp.Popen(['hadoop','fs','-mkdir', config['example_data']])
	subdirs = ["newtest", ]# "rotated",  "test",  "test-low"]
    for h in subdirs:
        for f in os.listdir(os.path.join(utmp, h)):
            fname =  os.path.join(utmp,h,f)
            hdfs_name = "%s_%s"%(h,f)
            sp.Popen(['hadoop', 'fs','-put',fname, 
                    os.path.join(config['example_data'], hdfs_name)]).communicate()
            if config.get('fuzzy_example_data', False):
                fuzzify(config, fname, hdfs_name)

