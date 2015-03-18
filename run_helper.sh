# These are some commands that helped with deployment when all files need to be sent

# where is image_analyzer locally
IMG=/Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/
export CLUSTER=image_cluster13

conda cluster manage $CLUSTER  install PIL  numpy
conda cluster manage $CLUSTER  install scikit-image
conda cluster manage $CLUSTER  install numpy scipy scikit-learn pandas flask
conda cluster submit $CLUSTER  $IMG/config.yaml --verbose
conda cluster submit $CLUSTER  $IMG/hdfs_paths.py --verbose
conda cluster submit $CLUSTER  $IMG/on_each_image.py --verbose
conda cluster submit $CLUSTER  $IMG/load_data.py --verbose
conda cluster submit $CLUSTER  $IMG/search.py --verbose
rmt (){
	conda cluster run.cmd $CLUSTER "$1"
}
rmt "rm -rf  /tmp/hdfs_tmp && mkdir -p  /tmp/hdfs_tmp && chown -R hdfs /tmp/hdfs_tmp"

conda cluster submit image_cluster13  $IMG/image_mapper.py --verbose
