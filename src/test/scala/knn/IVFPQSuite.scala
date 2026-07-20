package knn

import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

class IVFPQSuite extends AnyFunSuite {

  // Helper: brute-force exact KNN for recall comparison
  private def bruteForceKNN(
      query: Array[Float],
      database: Array[(Long, Array[Float])],
      k: Int
  ): Array[Long] = {
    database
      .map { case (id, vec) => (id, StandardMetric.Euclidean.distance(query, vec)) }
      .sortBy(_._2)
      .take(k)
      .map(_._1)
  }

  test("IVFPQIndex.train: produces valid codebook dimensions") {
    val rng = new Random(42)
    val dim = 16
    val numSubspaces = 4
    val data = Array.fill(500)(Array.fill(dim)(rng.nextFloat()))

    val codebook = IVFPQIndex.train(data, numCoarseCentroids = 8, numSubspaces = numSubspaces)

    assert(codebook.dimension === dim)
    assert(codebook.numSubspaces === numSubspaces)
    assert(codebook.coarseCentroids.length === 8)
    assert(codebook.coarseCentroids.forall(_.length === dim))
    assert(codebook.pqCentroids.length === numSubspaces)
    assert(codebook.pqCentroids.forall(_.length === 256))
    assert(codebook.pqCentroids.forall(_.forall(_.length === dim / numSubspaces)))
  }

  test("IVFPQIndex.encode: produces valid quantized record") {
    val rng = new Random(42)
    val dim = 16
    val numSubspaces = 4
    val data = Array.fill(500)(Array.fill(dim)(rng.nextFloat()))
    val codebook = IVFPQIndex.train(data, numCoarseCentroids = 8, numSubspaces = numSubspaces)

    val record = IVFPQIndex.encode(42L, data(0), codebook)
    assert(record.id === 42L)
    assert(record.coarseCentroidId >= 0 && record.coarseCentroidId < 8)
    assert(record.codes.length === numSubspaces)
  }

  test("IVFPQIndex.search: returns correct number of results") {
    val rng = new Random(42)
    val dim = 16
    val numSubspaces = 4
    val data = Array.fill(500)(Array.fill(dim)(rng.nextFloat()))
    val codebook = IVFPQIndex.train(data, numCoarseCentroids = 8, numSubspaces = numSubspaces)

    val database = data.zipWithIndex.map { case (vec, idx) =>
      IVFPQIndex.encode(idx.toLong, vec, codebook)
    }

    val query = data(0)
    val results = IVFPQIndex.search(query, k = 5, nprobe = 4, codebook, database)
    assert(results.length === 5)
    // Results should be sorted by distance ascending
    for (i <- 0 until results.length - 1) {
      assert(results(i).distance <= results(i + 1).distance)
    }
  }

  test("IVFPQIndex.search: recall@10 > 0.5 for random 16-dim data with nprobe=4") {
    val rng = new Random(42)
    val dim = 16
    val numSubspaces = 4
    val n = 1000
    val data = Array.fill(n)(Array.fill(dim)(rng.nextFloat()))
    val codebook = IVFPQIndex.train(data, numCoarseCentroids = 16, numSubspaces = numSubspaces)

    val database = data.zipWithIndex.map { case (vec, idx) =>
      IVFPQIndex.encode(idx.toLong, vec, codebook)
    }

    val databaseWithIds = data.zipWithIndex.map { case (vec, idx) => (idx.toLong, vec) }

    // Run 10 random queries and measure average recall
    val numQueries = 10
    val k = 10
    var totalRecall = 0.0

    for (q <- 0 until numQueries) {
      val query = data(rng.nextInt(n))
      val exactIds = bruteForceKNN(query, databaseWithIds, k).toSet
      val approxIds = IVFPQIndex.search(query, k, nprobe = 4, codebook, database).map(_.id).toSet
      val recall = exactIds.intersect(approxIds).size.toDouble / k
      totalRecall += recall
    }

    val avgRecall = totalRecall / numQueries
    assert(avgRecall > 0.5, s"Average recall@$k was $avgRecall, expected > 0.5")
  }

  test("IVFPQCodebook: JVM serialization round-trip") {
    val rng = new Random(42)
    val dim = 16
    val numSubspaces = 4
    val data = Array.fill(300)(Array.fill(dim)(rng.nextFloat()))
    val codebook = IVFPQIndex.train(data, numCoarseCentroids = 8, numSubspaces = numSubspaces)

    // Serialize
    val baos = new java.io.ByteArrayOutputStream()
    val oos = new java.io.ObjectOutputStream(baos)
    oos.writeObject(codebook)
    oos.close()
    val bytes = baos.toByteArray

    // Deserialize
    val bais = new java.io.ByteArrayInputStream(bytes)
    val ois = new java.io.ObjectInputStream(bais)
    val restored = ois.readObject().asInstanceOf[IVFPQCodebook]
    ois.close()

    assert(restored.dimension === codebook.dimension)
    assert(restored.numSubspaces === codebook.numSubspaces)
    assert(restored.coarseCentroids.length === codebook.coarseCentroids.length)
    assert(restored.metricName === codebook.metricName)

    // Verify the restored codebook works for encoding
    val record = IVFPQIndex.encode(1L, data(0), restored)
    assert(record.codes.length === numSubspaces)
  }
}
