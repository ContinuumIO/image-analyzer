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
import os
from StringIO import StringIO
utmp = "/tmp/hdfs_tmp/"
os.chdir(utmp)
TEST_DATA = 'http://vasc.ri.cmu.edu/idb/images/face/frontal_images/images.tar'
images_test = os.path.join(utmp, 'images_test')
def download_zipped_faces(config, url=TEST_DATA, fname=images_test):
    from tarfile import TarFile
    import os
    req = requests.get(url, stream=True)
    f = open(fname, 'w')
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            f.write(chunk)
            f.flush()
    with TarFile.open(fname) as tf:
    	tf.extractall()

	sp.Popen(['hadoop','fs','-mkdir','/imgs/'])
	subdirs = ["newtest",  "rotated",  "test",  "test-low"]
	for h in subdirs:
		for f in os.listdir(os.path.join(utmp, h)):
			f2 =  os.path.join(utmp,h,f)
			print(f2)
			sp.Popen(['hadoop', 'fs','-put',f2, os.path.join(config['example_data'], f)]).communicate()
	

