package knn

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.collection.mutable

object SparkIVFPQ {

  // Trains the IVFPQCodebook natively using Spark MLlib Distributed DataFrames
  def trainFromRDD(
      rdd: RDD[Array[Float]],
      numCoarseCentroids: Int,
      numSubspaces: Int,
      maxIterations: Int = 30,
      tolerance: Float = 1e-4f,
      metricName: String = "euclidean"
  ): IVFPQCodebook = {
    val sc = rdd.sparkContext
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()
    import spark.implicits._

    // Smart Dynamic Sampling Logic
    val totalCount = rdd.count()
    val sampleSize = if (totalCount <= 1000000L) {
      totalCount * 0.5
    } else {
      1000000.0 // Hardcap at 1M for safety
    }
    val fraction = math.min(1.0, sampleSize / totalCount.toDouble)

    println(s"Dynamic Sampling: Total Vectors = $totalCount. Sampling Fraction = $fraction (Target Size: ~${(totalCount * fraction).toInt})")

    // If Cosine, we normalize vectors before passing to KMeans
    val preparedRdd = if (metricName.toLowerCase == "cosine") {
      rdd.map(IVFPQIndex.normalize)
    } else {
      rdd
    }

    val sampleDf = preparedRdd
      .sample(withReplacement = false, fraction, seed = 42)
      .map(v => Tuple1(org.apache.spark.ml.linalg.Vectors.dense(v.map(f => f.toDouble))))
      .toDF("features")
      .cache() // Cache since we run multiple MLlib jobs on it

    val sampleCount = sampleDf.count()
    require(sampleCount > 0, "Sample dataset is empty. Increase input RDD size.")

    val dimension = rdd.first().length
    require(dimension % numSubspaces == 0, s"Dimension $dimension must be divisible by subspaces $numSubspaces")
    val subspaceDim = dimension / numSubspaces

    val finalNumCoarseCentroids = if (numCoarseCentroids > 0) numCoarseCentroids else math.max(1, math.min(4096, (totalCount / 3000).toInt))

    println(s"Step 1/3: Training Coarse Centroids (K=$finalNumCoarseCentroids) with Spark MLlib...")
    val coarseKMeans = new org.apache.spark.ml.clustering.KMeans()
      .setK(finalNumCoarseCentroids)
      .setMaxIter(maxIterations)
      .setTol(tolerance)
      .setFeaturesCol("features")

    val coarseModel = coarseKMeans.fit(sampleDf)
    val coarseCentroids = coarseModel.clusterCenters.map(v => v.toArray.map(_.toFloat))

    println("Step 2/3: Broadcasting coarse centroids to compute residuals...")
    val bcCoarseCentroids = sc.broadcast(coarseCentroids)

    // Calculate residuals distributedly
    val residualsRdd = sampleDf.rdd.map { row =>
      val sample = row.getAs[org.apache.spark.ml.linalg.Vector](0).toArray.map(_.toFloat)
      val localCoarse = bcCoarseCentroids.value
      
      // Find nearest coarse centroid
      var nearestIdx = 0
      var minDist = Float.MaxValue
      var c = 0
      while (c < finalNumCoarseCentroids) {
        val dist = Metric.Euclidean.distance(sample, localCoarse(c))
        if (dist < minDist) {
           minDist = dist
           nearestIdx = c
        }
        c += 1
      }
      
      // Calculate residual: r = sample - centroid
      val res = new Array[Float](dimension)
      var f = 0
      while (f < dimension) {
        res(f) = sample(f) - localCoarse(nearestIdx)(f)
        f += 1
      }
      res
    }.cache()
    
    // Force evaluation of residuals before we unpersist sampleDf
    residualsRdd.count()
    sampleDf.unpersist()

    println("Step 3/3: Training Product Quantization Subspaces with Spark MLlib...")
    val pqCentroids = Array.ofDim[Float](numSubspaces, 256, subspaceDim)

    var m = 0
    while (m < numSubspaces) {
      val startDim = m * subspaceDim
      
      // Map residual RDD into subspace slice DataFrame
      val subDf = residualsRdd.map { res =>
        val subVec = new Array[Double](subspaceDim)
        var s = 0
        while (s < subspaceDim) {
          subVec(s) = res(startDim + s).toDouble
          s += 1
        }
        Tuple1(org.apache.spark.ml.linalg.Vectors.dense(subVec))
      }.toDF("features").cache()

      println(s"  -> Training Subspace ${m + 1}/$numSubspaces (K=256)")
      val subKMeans = new org.apache.spark.ml.clustering.KMeans()
        .setK(256)
        .setMaxIter(15) // Fewer iterations for subspace codebooks
        .setTol(tolerance)
        .setFeaturesCol("features")

      val subModel = subKMeans.fit(subDf)
      pqCentroids(m) = subModel.clusterCenters.map(v => v.toArray.map(_.toFloat))
      
      subDf.unpersist()
      m += 1
    }

    residualsRdd.unpersist()

    IVFPQCodebook(dimension, numSubspaces, coarseCentroids, pqCentroids, metricName)
  }

  // Encodes raw vectors in an RDD into QuantizedRecords in parallel
  def indexRDD(
      rdd: RDD[(Any, Array[Float])],
      broadcastCodebook: Broadcast[IVFPQCodebook]
  ): RDD[QuantizedRecord] = {
    rdd.mapPartitions { iter =>
      val codebook = broadcastCodebook.value
      iter.map { case (id, vector) =>
        IVFPQIndex.encode(id, vector, codebook)
      }
    }
  }

  // Executes a distributed batch search using Asymmetric Distance Computation
  def distributedSearch(
      databaseRDD: RDD[QuantizedRecord],
      queries: Array[(Any, Array[Float])], // (queryId, queryVector)
      k: Int,
      nprobe: Int,
      broadcastCodebook: Broadcast[IVFPQCodebook]
  ): RDD[(Any, Array[SearchResult])] = {
    val sc = databaseRDD.sparkContext
    val broadcastQueries = sc.broadcast(queries)

    // Step 1: MapPartitions to find local top-k nearest neighbors on each partition
    val localTopK = databaseRDD.mapPartitions { recordIter =>
      val codebook = broadcastCodebook.value
      val localQueries = broadcastQueries.value
      val partitionRecords = recordIter.toArray

      if (partitionRecords.isEmpty) {
        Iterator.empty
      } else {
        localQueries.iterator.map { case (qId, qVec) =>
          val localResults = IVFPQIndex.search(qVec, k, nprobe, codebook, partitionRecords)
          (qId, localResults)
        }
      }
    }

    // Step 2: Group by queryId and merge local results into the global top-k
    localTopK.groupByKey().mapValues { localResultsIterable =>
      val pq = mutable.PriorityQueue.empty[SearchResult](Ordering.by(-_.distance))
      
      localResultsIterable.foreach { localResults =>
        var r = 0
        val len = localResults.length
        while (r < len) {
          val result = localResults(r)
          if (pq.size < k) {
            pq.enqueue(result)
          } else if (result.distance < pq.head.distance) {
            pq.dequeue()
            pq.enqueue(result)
          }
          r += 1
        }
      }
      
      pq.dequeueAll.reverse.toArray
    }
  }
}

/**
 * Fluent Builder API: Estimator for Training
 */
class SparkIVFPQ {
  private var numCoarseCentroids: Int = -1 // -1 means auto-scale based on dataset size
  private var numSubspaces: Int = 16
  private var maxIterations: Int = 30
  private var tolerance: Float = 1e-4f
  private var metricName: String = "euclidean"
  private var idCol: String = "id"

  def setNumCoarseCentroids(value: Int): this.type = { numCoarseCentroids = value; this }
  def setNumSubspaces(value: Int): this.type = { numSubspaces = value; this }
  def setMaxIterations(value: Int): this.type = { maxIterations = value; this }
  def setTolerance(value: Float): this.type = { tolerance = value; this }
  def setMetricName(value: String): this.type = { metricName = value; this }
  def setIdCol(value: String): this.type = { idCol = value; this }

  def fit(baseDf: DataFrame, vectorCol: String = "vector"): SparkIVFPQModel = {
    // Extract vectors from DataFrame
    val rawVectorsRdd = baseDf.select(vectorCol).rdd.map { row =>
      row.get(0) match {
        case v: org.apache.spark.ml.linalg.Vector => v.toArray.map(_.toFloat)
        case s: Seq[_] => s.map {
          case d: Double => d.toFloat
          case f: Float => f
          case i: Int => i.toFloat
        }.toArray
      }
    }

    val codebook = SparkIVFPQ.trainFromRDD(
      rawVectorsRdd,
      numCoarseCentroids,
      numSubspaces,
      maxIterations,
      tolerance,
      metricName
    )

    new SparkIVFPQModel(codebook).setIdCol(idCol)
  }
}

/**
 * Fluent Builder API: Model for Inference
 */
class SparkIVFPQModel(val codebook: IVFPQCodebook) {
  private var k: Int = 100
  private var kApprox: Option[Int] = None
  private var nprobe: Option[Int] = None
  private var idCol: String = "id"
  private var excludeSelf: Boolean = false

  def setK(value: Int): this.type = { k = value; this }
  def setKApprox(value: Int): this.type = { kApprox = Some(value); this }
  def setNprobe(value: Int): this.type = { nprobe = Some(value); this }
  def setIdCol(value: String): this.type = { idCol = value; this }
  def setExcludeSelf(value: Boolean): this.type = { excludeSelf = value; this }

  def transform(queryDf: DataFrame, baseDf: DataFrame, vectorCol: String = "vector"): DataFrame = {
    val spark = queryDf.sparkSession
    import spark.implicits._

    // Read base vectors along with their IDs (preserving datatype)
    val baseRdd = baseDf.select(idCol, vectorCol).rdd.map { row =>
      val id = row.get(0)
      val vec = row.get(1) match {
        case v: org.apache.spark.ml.linalg.Vector => v.toArray.map(_.toFloat)
        case s: Seq[_] => s.map {
          case d: Double => d.toFloat
          case f: Float => f
          case i: Int => i.toFloat
        }.toArray
      }
      (id, vec)
    }.cache()

    // Broadcast Codebook
    val broadcastCodebook = spark.sparkContext.broadcast(codebook)

    // Calculate Dynamic Heuristics for Inference
    val finalNprobe = nprobe.getOrElse(math.max(16, math.min(64, math.sqrt(codebook.coarseCentroids.length.toDouble).toInt)))
    // We cannot easily get dbSize without triggering a job, but we'll use a fast count
    val dbSize = baseRdd.count()
    val finalKApprox = kApprox.getOrElse(math.max(k * 20, (dbSize * 0.01).toInt))
    println(s"Dynamic Inference Heuristics -> nprobe: $finalNprobe, kApprox: $finalKApprox")

    // Extract query vectors (preserving datatype)
    val queryRdd = queryDf.select(idCol, vectorCol).rdd.map { row =>
      val id = row.get(0)
      val vec = row.get(1) match {
        case v: org.apache.spark.ml.linalg.Vector => v.toArray.map(_.toFloat)
        case s: Seq[_] => s.map {
          case d: Double => d.toFloat
          case f: Float => f
          case i: Int => i.toFloat
        }.toArray
      }
      (id, vec)
    }

    // Stage 1: Fast Approximate Search via Pure-Spark IVF CoGroup
    
    // 1a. Map Queries to their nearest `nprobe` coarse centroids
    val queryCentroidsRdd = queryRdd.flatMap { case (qId, qVec) =>
      val cb = broadcastCodebook.value
      val preparedQuery = if (cb.metricName.toLowerCase == "cosine") IVFPQIndex.normalize(qVec) else qVec
      
      val coarseDists = cb.coarseCentroids.zipWithIndex.map { case (cVec, cIdx) =>
        (cIdx, cb.getMetricImpl.distance(preparedQuery, cVec))
      }
      val topCoarse = coarseDists.sortBy(_._2).take(finalNprobe).map(_._1)
      topCoarse.map(cIdx => (cIdx, (qId, preparedQuery)))
    }

    // 1b. Encode database and key by coarseCentroidId
    val databaseRDD = SparkIVFPQ.indexRDD(baseRdd, broadcastCodebook)
    val dbCentroidsRdd = databaseRDD.map(record => (record.coarseCentroidId, record))

    // 1c. CoGroup and search locally per cluster (Zero database broadcast)
    val approxResultsRdd = queryCentroidsRdd.cogroup(dbCentroidsRdd).flatMap { case (cIdx, (queriesIter, recordsIter)) =>
      val cb = broadcastCodebook.value
      val records = recordsIter.toArray
      
      if (records.isEmpty) {
        Iterator.empty
      } else {
        queriesIter.map { case (qId, preparedQuery) =>
          val approxTopK = IVFPQIndex.searchInCluster(preparedQuery, cIdx, finalKApprox, cb, records)
          (qId, approxTopK)
        }
      }
    }

    // 1d. Global Top-K Reduction across clusters for each query
    val globalApproxTopK = approxResultsRdd.groupByKey().map { case (qId, resultsIter) =>
      val allResults = resultsIter.flatten.toArray
      val bestK = allResults.sortBy(_.distance).take(finalKApprox)
      (qId, bestK)
    }

    // Stage 2: Exact Shuffle Join Re-Ranking (Zero query broadcast)
    val explodedApprox = globalApproxTopK.flatMap { case (qId, results) =>
      results.map(res => (res.id, qId))
    }
    
    // Join with database to fetch raw vectors: nId -> (qId, dbVec)
    val approxWithDb = explodedApprox.join(baseRdd).map { case (nId, (qId, dbVec)) =>
      (qId, (nId, dbVec))
    }
    
    // Join with queries to fetch original query vectors: qId -> ((nId, dbVec), qVec)
    val exactDistances = approxWithDb.join(queryRdd).map { case (qId, ((nId, dbVec), qVec)) =>
      val metric = broadcastCodebook.value.getMetricImpl
      val exactDist = metric.distance(qVec, dbVec)
      (qId, SearchResult(nId, exactDist))
    }
    
    // Extract local variables to avoid capturing `this` in RDD closure (Task not serializable)
    val localK = k
    val localExcludeSelf = excludeSelf
    
    val exactResultsRdd = exactDistances.groupByKey().map { case (qId, results) =>
      val sorted = results.toArray.sortBy(_.distance)
      val finalTopK = if (localExcludeSelf) {
        sorted.filterNot(_.id == qId).take(localK)
      } else {
        sorted.take(localK)
      }
      (qId, finalTopK)
    }

    // Format as DataFrame: Arrays of neighbors and distances preserving the ID datatype
    val idType = queryDf.schema(idCol).dataType
    val outSchema = org.apache.spark.sql.types.StructType(Array(
      org.apache.spark.sql.types.StructField(idCol, idType, nullable = true),
      org.apache.spark.sql.types.StructField("approx_neighbors", org.apache.spark.sql.types.ArrayType(idType), nullable = true),
      org.apache.spark.sql.types.StructField("distances", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.DoubleType), nullable = true)
    ))

    import org.apache.spark.sql.Row
    val rowRdd = exactResultsRdd.map { case (qId, topK) =>
      val neighborIds = topK.map(_.id).toSeq
      val distances = topK.map(_.distance.toDouble).toSeq
      Row(qId, neighborIds, distances)
    }

    spark.createDataFrame(rowRdd, outSchema)
  }
}

