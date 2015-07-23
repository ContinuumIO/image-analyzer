# These are some commands that helped with deployment when all files need to be sent





rmt (){
	acluster cmd  "$1";
}
rmt_head (){
    acluster cmd "$1" -t head;
}
setup_remote (){
	acluster conda   install PIL  numpy;
	acluster conda   install scikit-image;
	acluster conda install  scipy scikit-learn pandas;
	rmt "apt-get install unzip";
}
# Run setup_remote just the first time
setup_remote

load_faces94 (){
    rmt "mkdir -p /tmp/anaconda-cluster; chown -R ubuntu /tmp/anaconda-cluster";
	acluster put   fuzzify_training.py /tmp/anaconda-cluster/fuzzify_training.py; 
	acluster put   hdfs_paths.py /tmp/anaconda-cluster/hdfs_paths.py;

	acluster put   load_faces94.sh /tmp/anaconda-cluster/load_faces94.sh;
	rmt_head "cd /tmp/anaconda-cluster; source load_faces94.sh" ;
}

# Run this once if you want the faces94
# dataset.  You can interrupt it if you 
# get tired of waiting and use the photos loaded so far.
load_faces94


# Run each of these file submit commands
# on first time or if you change the files.
# Note you will see an error when submitting the config.yaml
# because it tries to run the .yaml as python script.  
# Change these command to use "put" when 
# that command is available.

acluster put   config.yaml /tmp/anaconda-cluster/config.yaml 
acluster put   map_each_image.py /tmp/anaconda-cluster/map_each_image.py 
acluster put   hdfs_paths.py /tmp/anaconda-cluster/hdfs_paths.py
acluster put   search.py  /tmp/anaconda-cluster/search.py



# Finally, running it, referencing the files above, and 
# using the settings in config.yaml.

acluster submit --stream  image_mapper.py 
