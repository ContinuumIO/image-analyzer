#!/bin/bash
set -e 
set -x
rm -rf /tmp/anaconda-cluster/hdfs_tmp
mkdir -p  /tmp/anaconda-cluster/hdfs_tmp
chown -R hdfs /tmp/anaconda-cluster/hdfs_tmp
cd /tmp/anaconda-cluster/hdfs_tmp
faces94="http://cswww.essex.ac.uk/mv/allfaces/faces94.zip"
wget ${faces94}
unzip faces94
cd faces94
hadoop fs -mkdir -p /fuzzy
hadoop fs -mkdir -p /imgs 
for line in $(find ./ | sort -R | grep jpg$ );
	do 
	# clean out slashes and dots
	no_slash=$(echo $line | sed 's/\///g' | sed 's/\._//g' | sed 's/^\.//');
	hadoop fs -put ${line} /imgs/$no_slash;
	python /tmp/anaconda-cluster/fuzzify_training.py $line t1 /fuzzy/ ${no_slash};
done

