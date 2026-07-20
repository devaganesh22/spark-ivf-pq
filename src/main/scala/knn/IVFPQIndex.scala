package knn

import java.io._
import scala.collection.mutable
import scala.util.Random

// Represents the serializable codebooks containing trained centroids
case class IVFPQCodebook(
    dimension: Int,
    numSubspaces: Int,
    coarseCentroids: Array[Array[Float]], // Size: C x D
    pqCentroids: Array[Array[Array[Float]]], // Size: M x 256 x (D/M)
    metricName: String
) extends Serializable {
  
  @transient private var _metricImpl: Metric[Array[Float]] = null

  def getMetricImpl: Metric[Array[Float]] = {
    if (_metricImpl == null) {
      _metricImpl = metricName.toLowerCase match {
        case "cosine" => Metric.Cosine
        case _ => Metric.Euclidean
      }
    }
    _metricImpl
  }
}

object IVFPQCodebook {
  def save(codebook: IVFPQCodebook, path: String): Unit = {
    val fileOut = new java.io.FileOutputStream(path)
    val out = new java.io.ObjectOutputStream(fileOut)
    try {
      out.writeObject(codebook)
    } finally {
      out.close()
      fileOut.close()
    }
  }

  def load(path: String): IVFPQCodebook = {
    val fileIn = new java.io.FileInputStream(path)
    val in = new java.io.ObjectInputStream(fileIn)
    try {
      in.readObject().asInstanceOf[IVFPQCodebook]
    } finally {
      in.close()
      fileIn.close()
    }
  }
}

// Represents a quantized vector in the index
case class QuantizedRecord(
    id: Long,
    coarseCentroidId: Int,
    codes: Array[Byte] // Size: M
) extends Serializable

// Represents a nearest neighbor search result
case class SearchResult(
    id: Long,
    distance: Float
) extends Serializable

object IVFPQIndex {

  // Normalizes a vector to unit length (L2 norm = 1) for Cosine distance
  def normalize(v: Array[Float]): Array[Float] = {
    var sum = 0.0f
    var i = 0
    val len = v.length
    while (i < len) {
      sum += v(i) * v(i)
      i += 1
    }
    val norm = math.sqrt(sum).toFloat
    if (norm == 0.0f) v.clone()
    else {
      val res = new Array[Float](len)
      i = 0
      while (i < len) {
        res(i) = v(i) / norm
        i += 1
      }
      res
    }
  }



  // Encodes a raw vector into a quantized record
  def encode(
      id: Long,
      vector: Array[Float],
      codebook: IVFPQCodebook
  ): QuantizedRecord = {
    val preparedVector = if (codebook.metricName.toLowerCase == "cosine") {
      normalize(vector)
    } else {
      vector
    }

    val dimension = codebook.dimension
    val numSubspaces = codebook.numSubspaces
    val subspaceDim = dimension / numSubspaces

    // 1. Find closest coarse centroid
    var coarseCentroidId = 0
    var minDist = Float.MaxValue
    var c = 0
    val numCoarse = codebook.coarseCentroids.length
    while (c < numCoarse) {
      val dist = Metric.Euclidean.distance(preparedVector, codebook.coarseCentroids(c))
      if (dist < minDist) {
        minDist = dist
        coarseCentroidId = c
      }
      c += 1
    }

    // 2. Compute residual
    val residual = new Array[Float](dimension)
    var f = 0
    while (f < dimension) {
      residual(f) = preparedVector(f) - codebook.coarseCentroids(coarseCentroidId)(f)
      f += 1
    }

    // 3. Quantize residual into codes
    val codes = new Array[Byte](numSubspaces)
    var m = 0
    val subVec = new Array[Float](subspaceDim)
    while (m < numSubspaces) {
      val startDim = m * subspaceDim
      System.arraycopy(residual, startDim, subVec, 0, subspaceDim)

      // Find nearest PQ centroid in subspace m
      var nearestPqIdx = 0
      var minPqDist = Float.MaxValue
      var k = 0
      while (k < 256) {
        val dist = Metric.Euclidean.distance(subVec, codebook.pqCentroids(m)(k))
        if (dist < minPqDist) {
          minPqDist = dist
          nearestPqIdx = k
        }
        k += 1
      }
      codes(m) = nearestPqIdx.toByte
      m += 1
    }

    QuantizedRecord(id, coarseCentroidId, codes)
  }

  // Local/standalone search execution using ADC (Asymmetric Distance Computation)
  def search(
      query: Array[Float],
      k: Int,
      nprobe: Int,
      codebook: IVFPQCodebook,
      database: Array[QuantizedRecord]
  ): Array[SearchResult] = {
    // 1. Group database records into buckets (inverted lists)
    val numCoarse = codebook.coarseCentroids.length
    val buckets = Array.fill(numCoarse)(mutable.ArrayBuffer.empty[QuantizedRecord])
    var i = 0
    val dbSize = database.length
    while (i < dbSize) {
      val rec = database(i)
      buckets(rec.coarseCentroidId).append(rec)
      i += 1
    }

    // 2. Find nearest coarse centroids to the query
    val preparedQuery = if (codebook.metricName.toLowerCase == "cosine") {
      normalize(query)
    } else {
      query
    }

    val coarseDistances = codebook.coarseCentroids.zipWithIndex.map { case (centroid, idx) =>
      val dist = Metric.Euclidean.distance(preparedQuery, centroid)
      (idx, dist)
    }.sortBy(_._2)

    val probes = coarseDistances.take(nprobe).map(_._1)

    // Max heap to track top-k nearest neighbors
    val pq = mutable.PriorityQueue.empty[SearchResult](Ordering.by(_.distance))

    val dimension = codebook.dimension
    val numSubspaces = codebook.numSubspaces
    val subspaceDim = dimension / numSubspaces

    // 3. For each coarse centroid probe, build lookup table and scan records
    var p = 0
    val probeCount = probes.length
    val querySubVec = new Array[Float](subspaceDim)

    while (p < probeCount) {
      val coarseCentroidId = probes(p)
      val coarseCentroid = codebook.coarseCentroids(coarseCentroidId)
      val records = buckets(coarseCentroidId)

      if (records.nonEmpty) {
        // Compute query residual for this coarse cluster
        val queryResidual = new Array[Float](dimension)
        var f = 0
        while (f < dimension) {
          queryResidual(f) = preparedQuery(f) - coarseCentroid(f)
          f += 1
        }

        // Build lookup table of size M x 256: query-to-PQ-centroids distances
        val lookupTable = Array.ofDim[Float](numSubspaces, 256)
        var m = 0
        while (m < numSubspaces) {
          val startDim = m * subspaceDim
          System.arraycopy(queryResidual, startDim, querySubVec, 0, subspaceDim)

          var kCent = 0
          while (kCent < 256) {
            val dist = Metric.Euclidean.distance(querySubVec, codebook.pqCentroids(m)(kCent))
            // Store distance squared to speed up calculations (avoiding sqrt if possible, 
            // but since Metric.distance returns actual distance, we compute distance squared for ADC search)
            lookupTable(m)(kCent) = dist * dist
            kCent += 1
          }
          m += 1
        }

        // Scan records in the bucket using lookup table (ADC)
        var r = 0
        val recCount = records.length
        while (r < recCount) {
          val rec = records(r)
          var distSq = 0.0f
          var sub = 0
          while (sub < numSubspaces) {
            val codeIdx = rec.codes(sub) & 0xFF
            distSq += lookupTable(sub)(codeIdx)
            sub += 1
          }

          // Map distance squared back to final search distance
          val finalDistance = if (codebook.metricName.toLowerCase == "cosine") {
            // For Cosine, we used normalized vectors. Cosine distance = 1 - cos(theta) = L2_dist_sq / 2.
            distSq / 2.0f
          } else {
            math.sqrt(distSq).toFloat
          }

          val result = SearchResult(rec.id, finalDistance)

          if (pq.size < k) {
            pq.enqueue(result)
          } else if (finalDistance < pq.head.distance) {
            pq.dequeue()
            pq.enqueue(result)
          }
          r += 1
        }
      }
      p += 1
    }

    pq.dequeueAll.reverse.toArray
  }
}
