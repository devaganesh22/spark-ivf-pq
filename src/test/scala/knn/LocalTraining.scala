package knn

import scala.util.Random

object LocalTraining {

  def localKMeans(
      data: Array[Array[Float]],
      k: Int,
      maxIterations: Int,
      seed: Long = 123L
  ): Array[Array[Float]] = {
    if (data.isEmpty) return Array.empty
    val dim = data.head.length
    // Initialize centroids by evenly sampling from data
    val centroids = Array.tabulate(k)(i => data((i * data.length) / k).clone())

    var iter = 0
    var changed = true
    while (iter < maxIterations && changed) {
      val assignments = data.map { vec =>
        var minCentroid = 0
        var minDist = Float.MaxValue
        var i = 0
        while (i < k) {
          val dist = StandardMetric.Euclidean.distance(vec, centroids(i))
          if (dist < minDist) {
            minDist = dist
            minCentroid = i
          }
          i += 1
        }
        minCentroid
      }

      val newCentroids = Array.fill(k)(new Array[Float](dim))
      val counts = new Array[Int](k)

      var i = 0
      while (i < data.length) {
        val cluster = assignments(i)
        val vec = data(i)
        var d = 0
        while (d < dim) {
          newCentroids(cluster)(d) += vec(d)
          d += 1
        }
        counts(cluster) += 1
        i += 1
      }

      changed = false
      var c = 0
      while (c < k) {
        if (counts(c) > 0) {
          var d = 0
          while (d < dim) {
            newCentroids(c)(d) /= counts(c)
            if (math.abs(newCentroids(c)(d) - centroids(c)(d)) > 1e-4) {
              changed = true
            }
            d += 1
          }
          centroids(c) = newCentroids(c)
        }
        c += 1
      }
      iter += 1
    }
    centroids
  }

  def localTrain(
      data: Array[Array[Float]],
      numCoarseCentroids: Int,
      numSubspaces: Int
  ): IVFPQCodebook = {
    val dim = data.head.length
    val coarseCentroids = localKMeans(data, numCoarseCentroids, 20)

    val residuals = data.map { vec =>
      var minCentroid = 0
      var minDist = Float.MaxValue
      for (i <- coarseCentroids.indices) {
        val dist = StandardMetric.Euclidean.distance(vec, coarseCentroids(i))
        if (dist < minDist) {
          minDist = dist
          minCentroid = i
        }
      }
      val res = new Array[Float](dim)
      for (d <- 0 until dim) res(d) = vec(d) - coarseCentroids(minCentroid)(d)
      res
    }

    val pqCentroids = Array.ofDim[Float](numSubspaces, 256, dim / numSubspaces)
    val subspaceDim = dim / numSubspaces

    for (m <- 0 until numSubspaces) {
      val startDim = m * subspaceDim
      val subData = residuals.map { res =>
        val subVec = new Array[Float](subspaceDim)
        System.arraycopy(res, startDim, subVec, 0, subspaceDim)
        subVec
      }
      pqCentroids(m) = localKMeans(subData, 256, 15)
    }

    IVFPQCodebook(dim, numSubspaces, coarseCentroids, pqCentroids, "euclidean")
  }
}
