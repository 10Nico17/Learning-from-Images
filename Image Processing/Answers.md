Exercise 2: K-Means 

a) What are the problems of this clustering algorithm?
- slow, needs a long time to compute (images with high resolution and many pixel)
- the result depends on the random initialization of the means at the beginning. You get different results for the same image.
- we don't know what is a good value at the beginning for number of clusters

b) How can I improve the results?
- K-Medians instead of K-Means: Instead of using the average (mean) as the cluster center, using the median can provide better robustness against outliers. 
- use librarys with optimized (faster) code 
- instead of random initialization, you can use more advanced methods -> Initially select centers that are far apart from each other --> (this is implemented if you activate it)
- use methods such as the elbow method to determine the optimal number of clusters 
