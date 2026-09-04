# K-Means Clustering from Scratch

A basic implementation of the K-Means clustering algorithm in C++, built from scratch without using ML libraries.

## Includes

- `<iostream>`
- `<vector>`
- `<cmath>`
- `<limits>`
- `<stdexcept>`
- `<random>`
- `<algorithm>`

## Features

- Random K-centroid initialization
- Euclidean distance
- Cluster assignment
- Centroid updating
- Convergence checking
- Maximum iteration limit
- Empty-cluster handling
- Input validation
- `predict()` for new points
- Encapsulated class design

## Working

1. Validate the input dataset
2. Select K random points as initial centroids
3. Assign each point to its nearest centroid
4. Calculate new centroids
5. Check if centroids have converged
6. Repeat until convergence or maximum iterations
7. Use `predict()` to assign new points to a cluster

## Purpose

Built from scratch to understand K-Means clustering and C++ class design.