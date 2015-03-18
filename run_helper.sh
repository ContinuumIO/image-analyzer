#conda cluster manage image_cluster13  install PIL  numpy
#conda cluster manage image_cluster13  install scikit-image
#conda cluster manage image_cluster13  install numpy scipy scikit-learn pandas flask

conda cluster submit image_cluster13  /Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/config.yaml --verbose
#conda cluster submit image_cluster13  /Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/on_each_image.py --verbose
#conda cluster submit image_cluster13  /Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/load_data.py --verbose
#conda cluster submit image_cluster13  /Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/search.py --verbose
export CLUSTER=image_cluster13
rmt (){
	conda cluster run.cmd $CLUSTER "$1"
}
rmt "rm -rf  /tmp/hdfs_tmp && mkdir -p  /tmp/hdfs_tmp && chown -R hdfs /tmp/hdfs_tmp"

conda cluster submit image_cluster13  /Users/petersteinberg/Desktop/PeterCode/continuum/image-analyzer/image_mapper.py --verbose