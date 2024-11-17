# Homework 4 - Clustering and Dimensionality Reduction

This repository contains implementations and analyses of **Hierarchical Clustering**, **K-medians Clustering**, and **Principal Components Analysis (PCA)**. The homework emphasizes both theoretical understanding and practical applications, covering clustering methods based on different linkage criteria, robust clustering with K-medians, and dimensionality reduction using PCA.

## Summary

This homework covered key clustering and dimensionality reduction techniques:

### 1. Hierarchical Clustering
   - Explored **Single-Link (MIN)** and **Complete-Link (MAX)** clustering algorithms, demonstrating their iterative merging processes using a similarity matrix.
   - Constructed dendrograms for each approach and analyzed the clustering structures formed when using different linkage criteria.
   - Created 3 clusters by selecting a cut on the dendrogram, allowing for visual and analytical insights into hierarchical clustering.

### 2. K-medians Clustering
   - Presented the K-medians algorithm as a variation of K-means, utilizing **Manhattan distance** and **medians** to determine cluster centroids, making it more robust to outliers.
   - Explained how the median is calculated for each cluster dimension and justified the effectiveness of K-medians in reducing the impact of outliers compared to K-means.

### 3. Principal Components Analysis (PCA)
   - Implemented PCA on a set of three 2D data points without using eigendecomposition, detailing each step from centering data to finding the principal component.
   - Projected data onto the first principal component to reduce it to a 1-dimensional representation, demonstrating dimensionality reduction and calculating new coordinates.

## Key Highlights
- **Hierarchical Clustering with Dendrograms**: Visualized hierarchical clustering with dendrograms, explaining how different linkage criteria (MIN vs. MAX) impact cluster formation.
- **Clustering with K-medians**: Demonstrated robust clustering with K-medians, reducing outlier impact with median-based centroids.
- **Dimensionality Reduction via PCA**: Projected data into 1-D space using PCA, illustrating core principles of dimensionality reduction without eigendecomposition.

---

## Full Report

For a detailed breakdown of derivations, algorithms, and results, please refer to the full report in the [PDF file](report/Homework4_Report.pdf).
