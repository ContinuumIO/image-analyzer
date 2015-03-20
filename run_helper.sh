# These are some commands that helped with deployment when all files need to be sent


# Parameters to set:
# Where is image_analyzer locally
IMG=/Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer
# what is the cluster name
export CLUSTER=image_cluster13
# end of parameters



rmt (){
	conda cluster run.cmd $CLUSTER "$1";
}
setup_remote (){
	conda cluster manage $CLUSTER  install PIL  numpy;
	conda cluster manage $CLUSTER  install scikit-image;
	conda cluster manage $CLUSTER  install numpy scipy scikit-learn pandas tornado;
}
# Run setup_remote just the first time
setup_remote

# Run each of these on first time or if you change them
conda cluster submit $CLUSTER  $IMG/config.yaml 
conda cluster submit $CLUSTER  $IMG/hdfs_paths.py --verbose
conda cluster submit $CLUSTER  $IMG/fuzzify_training.py --verbose
conda cluster submit $CLUSTER  $IMG/map_each_image.py --verbose

conda cluster submit $CLUSTER  $IMG/search.py --verbose
# the following is for a temporary directory into which
# hdfs user can download and untar an example images tar
rmt "rm -rf  /tmp/hdfs_tmp && mkdir -p  /tmp/hdfs_tmp && chown -R hdfs /tmp/hdfs_tmp"
conda cluster submit $CLUSTER  $IMG/load_data.py --verbose
# Finally, running it, referencing the files above, and 
# using the settings in config.yaml.
conda cluster submit image_cluster13  $IMG/image_mapper.py --verbose

