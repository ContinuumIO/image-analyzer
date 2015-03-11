from os.path import realpath
import sys
import numpy as np
from numpy.random import rand
from numpy import matrix
from pyspark import SparkContext
import time
from .on_each_image  import on_each_image



all_start = time.time()


if __name__ == "__main__":
    """
    Usage: image_mapper.py images_hdfs partitions
    """


    images_hdfs = sys.argv[0]  
    partitions = int(sys.argv[2]) 
    on_each_image(image_object=None, img_name=None)

    sc = SparkContext(appName="PythonALS")
    imgs_to_process = sc.textFile(images_hdfs)
    img = sc.parallelize(enumerate(imgs_to_process), partitions) \
               .map(lambda x: on_each_image(x)) \
               .collect()
