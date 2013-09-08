kmeans-mr
=========

 K-Means clustering demo code
 
 Generate data using DataGenerator and then run KMeansClusteringDriver.
 
 This driver iterates over the data to improve the cluster centroids until an
 iteration cutoff or converge
  
 NOTE: the data generator is only making a kind of gaussian cloud around a few
 centroids. This is very easy to solve by kmeans. To solve more difficult
 problems, I'd recommend using a professionnal implementation such as Apache
 Mahout
  
 Based in part on: <a href="http://codingwiththomas.blogspot.ca/2011/05/k-means-clustering-with-mapreduce.html">Thomas Jungblut's blog post</a>
 