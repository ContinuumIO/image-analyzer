## image-analyzer
### Steps in Analysis on Each Image
* Standardize image (resizing, removal of alpha channel for now)
* Percentiles of colors in standardized image.  
* Kmeans standardized image. Output the image's centroids.
* Perceptive hash of standardized image.  Encoded as hex and in chunks.
* PCA factors and variance.

### On my Mac, each image takes about .2 (mean) +/- .3 (stdev) seconds to process (skewed right).

### TODO on each image analysis 
* Experiment with numba for patch statistics 
* Standardize images better (padding is not done)
* Add a job group to different parts of the process so they can be killed if needed
* Add the scikit learn's agglomerative clustering feature recognition to on_each_image.py (hash those features)
* Could the same on_each_image function be applied again to all photos but zooming in or as a tiny thumbnail?
* When searching for matches of candidate images against db, save any ancillary information learned

### Pipeline
* Map the images from spark/hdfs to on_each_image function
* Output as example below to a table so that the outer machine learning algorithm can revisit image results without recalculating them.
* Do kmeans where the columns are the histogram and centroids of each image
* Keep a bag of perceptive hash chunks within each cluster
* Make inverse map tables like perceptive hash key to cluster id or hash to picture id
* Extract covariance matrix in kmeans passes
### Example output for each image
<code>
{
 'histo': array([[ 0.30574449,  0.48011642,  0.6848652 ,  0.99852941,  1.        ],
        [ 0.33168199,  0.48903186,  0.69650735,  1.        ,  1.        ],
        [ 0.29589461,  0.45309436,  0.6689951 ,  0.99215686,  0.99215686]]),
        
 'phash': ['0xfffe00000000fffe00000000fffe00000000ffff80200000ffff00000000',
  '0x80000000fffe00000000fffe20000000fffe20000000fffe00000000fffe0001',
  '0xfffe01000000ffff81000000ffff00000000fffe00000000fffe00000000ffff',
  '0xffff20000000ffff80000000ffff80000000ffff80000000ffff80000000',
  '0x80000000ffff00000000ffff00000000ffff00000000ffff08000000ffffa801',
  '0xffffe0000000ffffa0000000ffff80000000ffff80000000ffff000000010000',
  '0xffffc0000000ffffc4000000ffffc4000000ffffe0000000ffffe0000000',
  '0xe05cc040fffff00ffa00ffffc0000400ffffe0000000ffffe4000000ffffe001',
  '0xffffff3e57fffffffbfe07ffffffe0ffffffffffe00e77ffffffc00003f90000',
  '0xe3ffffffffffc1fffffff1ffe0ffffffffff043fffffffff27c7fffffffe5800',
  '0x1fffff5f83f803fffffce0f8f068fffffc3ffe1ff2ffffff3fcffaffffffff'],
  
 'cen_3': array([ 0.82521313,  0.85179961,  0.83754307]),
 'cen_2': array([ 0.54972566,  0.54918236,  0.51381149]),
 'cen_1': array([ 0.43394845,  0.45123864,  0.4188573 ]),
 'cen_0': array([ 0.98825424,  0.99303591,  0.98463453]),
 'cen_6': array([ 0.67782856,  0.68964991,  0.66018018]),
 'cen_5': array([ 0.18055515,  0.20905392,  0.20022794]),
 'cen_4': array([ 0.3263156 ,  0.35111616,  0.3175155 ]),
 
 'pca_var': array([  2.09597536e-01,   3.30507631e-04,   5.24692794e-05]),
 
 'pca_fac': array([[ 0.57727933,  0.56610672,  0.5884486 ],
        [-0.77360623,  0.14855585,  0.61600695],
        [ 0.26130819, -0.81083559,  0.5237019 ]])}
        
</code>
