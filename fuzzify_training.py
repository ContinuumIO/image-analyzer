from __future__ import division, print_function
import subprocess as sp
import numpy as np
import os
from hdfs_paths import hdfs_path
def fuzzify(config, fname, hdfs_name):
	from PIL import Image
	img = Image.open(fname)
	n = np.array(img)
	for _ in range(10):
		n = n +  sum(np.gradient(n)) * .001
	new = Image.fromarray(np.array(n, dtype="uint8"))
	loc_name = fname + 'fuz'
	new.save(loc_name,format="png")
	print(sp.Popen(['hadoop', 
				'fs',
				'-put', 
				loc_name, 
				hdfs_path(config, 
						config['fuzzy_example_data'], 
						hdfs_name)]))