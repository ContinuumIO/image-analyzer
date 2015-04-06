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
	rmt "apt-get install unzip";
}
# Run setup_remote just the first time
setup_remote

load_faces94 (){
	conda cluster submit $CLUSTER  $IMG/fuzzify_training.py --verbose
	conda cluster submit $CLUSTER  $IMG/hdfs_paths.py --verbose

	conda cluster submit $CLUSTER  $IMG/load_faces94.sh --verbose;
	rmt "cd /tmp; source load_faces94.sh" ;
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

conda cluster submit $CLUSTER  $IMG/config.yaml 
conda cluster submit $CLUSTER  $IMG/map_each_image.py --verbose
conda cluster submit $CLUSTER  $IMG/hdfs_paths.py --verbose
conda cluster submit $CLUSTER  $IMG/search.py --verbose



# Finally, running it, referencing the files above, and 
# using the settings in config.yaml.

conda cluster submit $CLUSTER  $IMG/image_mapper.py --verbose
