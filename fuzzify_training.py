from __future__ import division, print_function
import subprocess as sp
from scipy.ndimage.filters import laplace
import numpy as np
import os
import random
from hdfs_paths import hdfs_path
filterx = filtery = 6
change_perc = 0.14
def fuzzify(config, fname, hdfs_name):
    from PIL import Image
    img = Image.open(fname)
    n = np.array(img)
    for i in range(0,n.shape[0] -filterx, filterx):
        for j in range(0,n.shape[1]-filtery, filtery):
            if random.uniform(0, 1) < change_perc:
                for z in range(3):
                    n[i: i+filterx, j:j+filtery,z] = np.median(n[i:i+filterx,j:j+filtery,z])
    new = Image.fromarray(np.array(np.round(n), dtype=np.uint8))
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