from __future__ import division, print_function
import subprocess as sp
import numpy as np
import os
from hdfs_paths import hdfs_path
def fuzzify(config, fname, hdfs_name):
	from PIL import Image
	img = Image.open(fname)
	n = np.array(img)
	for _ in range(25):
		n = n +  sum(np.gradient(n)) * .005
	new = Image.fromarray(np.array(round(n), dtype=np.uint8))
	loc_name = fname + 'fuz'
	new.save(loc_name,format="png")
	print(sp.Popen(['hadoop', 
				'fs',
				'-put', 
				loc_name, 
				hdfs_path(config, 
						config['fuzzy_example_data'], 
						hdfs_name)]).communicate())


if __name__ == "__main__":
	import sys
	local_name, test_name, fuzzy_example_data, hdfs_name = sys.argv[1:]
	fuzzify({'test_name':test_name,
			'fuzzy_example_data': fuzzy_example_data}, 
			local_name, hdfs_name)