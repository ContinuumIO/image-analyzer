from __future__ import division, print_function
import subprocess as sp
from hdfs_paths import hdfs_path
def fuzzify(fname, config):
	from PIL import Image
	img = Image.open(fname)
	n = np.array(img)
	for _ in range(10):
		n =n +  sum(np.gradient(n)) * .001
	new = Image.fromarray(np.array(n, dtype="uint8"))
	loc_name = 'fuz_' + fname
	new.save(loc_name,format="png")
	sp.Popen(['hadoop', 
				'fs',
				'-put', 
				loc_name, 
				hdfs_paths(config['fuzzy_example_data'], 
					fname)])