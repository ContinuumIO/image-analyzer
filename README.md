## image-analyzer
### Steps in Analysis on Each Image
* Standardize image (resizing, removal of alpha channel for now)
* Percentiles of colors in standardized image.  
* Kmeans of the colors within 1 standardized image. Output the image's centroids.
* Perceptive hash of standardized image.  Hashed in chunks.
* PCA factors and variance.
* Ward clustering <a>http://scikit-learn.org/stable/auto_examples/cluster/plot_lena_ward_segmentation.html>as described here </a>

* Images are analyzed in full and in quadrants

### Timing
* On my Mac, each image takes about .2 (mean) +/- .3 (stdev) seconds to process (skewed right).  Depends on resolutions settings in config.yaml.

### TODO on each image analysis 
* Speed up loading of images to hdfs ?
* Experiment with numba for patch statistics 
* Standardize images better (padding is not done)
* Add a job group to different parts of the process so they can be killed if needed
* When searching for matches of candidate images against db, save any ancillary information learned
* Clean up system of hdfs pathing so that it is more modular (specific paths can be added to searches, etc)
* As part of that hdfs path clean up, also do some checks for required paths earlier on in algorithms
* Analysis of image metadata is not done, but some of the PIL Image data are saved in dict in map_each_image
* The config.yaml allows for patches to be specified.  The steps in find_similar involving joins on hashes are becoming slow when patches are included.  It may be necessary to break up the patch images from the rest of the analysis to use as a last resort.

### Pipeline Steps
* Map the images from spark/hdfs to on_each_image function
* Output as example below to a table so that the outer machine learning algorithm can revisit image results without recalculating them.
* Kmeans among all images where the columns are the histogram and centroids of each image's colors alone
* During kmeans, keep counts of common perceptive hash chunks and common ward cluster hashes
* Make inverse map tables like perceptive hash key to cluster id and ward cluster hash to image id


### HDFS Directory structure
* input_spec: where in hdfs are the training images, e.g. /imgs/*
* Test directory is based on given test name, like t1:

<code>
  t1/
    
    map_each_image/
    
      measures
    
    candidates/
      
      c1/
      
        measures
    
      c2/
      
        measures
    km/
    
      cluster_center_meta
      
      phash_unions
      
      ward_unions
      
      cluster_to_flattened
      
      cluster_to_key
      
      cluster_to_phash
      
      cluster_to_ward
      
      flattened_to_cluster
      
      flattened_to_key
      
      flattened_to_phash
      
      key_to_cluster
      
      key_to_phash
      
      phash_to_cluster
      
      phash_to_flattened
      
      phash_to_key
      
      ward_to_cluster
      
      ward_to_key

</code>


### Example output for each image
<code>
{
 'histo': array([[ 0.30574449,  0.48011642,  0.6848652 ,  0.99852941,  1.        ],
        [ 0.33168199,  0.48903186,  0.69650735,  1.        ,  1.        ],
        [ 0.29589461,  0.45309436,  0.6689951 ,  0.99215686,  0.99215686]]),
        
'phash': [        '0x100000000',
                  '0x100000000',
                  '0x100000000',
                  '0x100000000',
        ....
        ....
        ....
                  '0x100000000',
                  '0x100000000',
                  '0x100000000',
                  '0x76bf971f',
                  '0xb77ff7f0',
                  '0xff7ffce0',
                  '0xfff3fef8'],

 'cen': array([17, 17, 17, 48, 48, 48,  3,  3,  3, 58, 58, 58, 37, 37, 37,  9,  9,
        9, 27, 27, 27], dtype=int32),
 
'ward': (8798385704298443638,
                           -2098177597065484460,
                           -49937642176542373,
                           7306362158214069439,
                           -2098177597065484460,
                           1675627364594718983,
                           -2098177597065484460,
                           -5434838886404571572,
                           7306362158214069439,
                           -4035475343318357777,
                           5239582709862753648,
                           502896730504143507,
                           -5434838886404571572,
                           -5434838886404571572,
                           -6574300014753425568,
                           -2213663990102495809,
                           -3213379917254413273,)

 'pca_var': array([  2.09597536e-01,   3.30507631e-04,   5.24692794e-05]),
 
 'pca_fac': array([[ 0.57727933,  0.56610672,  0.5884486 ],
        [-0.77360623,  0.14855585,  0.61600695],
        [ 0.26130819, -0.81083559,  0.5237019 ]])}
        
</code>
