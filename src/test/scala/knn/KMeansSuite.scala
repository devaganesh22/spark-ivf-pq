package knn

import org.scalatest.funsuite.AnyFunSuite
import scala.util.Random

class KMeansSuite extends AnyFunSuite {

  test("KMeans.fit: produces the correct number of centroids") {
    val rng = new Random(123)
    val data = Array.fill(500)(Array.fill(8)(rng.nextFloat()))
    val centroids = LocalTraining.localKMeans(data, k = 10, maxIterations = 20)
    assert(centroids.length === 10)
    assert(centroids.forall(_.length === 8))
  }

  test("KMeans.fit: centroids converge to cluster centers for well-separated data") {
    // Generate 3 tight clusters centered at (0,0), (10,0), (0,10)
    val rng = new Random(42)
    val cluster1 = Array.fill(100)(Array(0.0f + rng.nextFloat() * 0.1f, 0.0f + rng.nextFloat() * 0.1f))
    val cluster2 = Array.fill(100)(Array(10.0f + rng.nextFloat() * 0.1f, 0.0f + rng.nextFloat() * 0.1f))
    val cluster3 = Array.fill(100)(Array(0.0f + rng.nextFloat() * 0.1f, 10.0f + rng.nextFloat() * 0.1f))
    val data = cluster1 ++ cluster2 ++ cluster3

    val centroids = LocalTraining.localKMeans(data, k = 3, maxIterations = 100, seed = 123L)

    // Each centroid should be close to one of the cluster centers
    val expectedCenters = Array(Array(0.0f, 0.0f), Array(10.0f, 0.0f), Array(0.0f, 10.0f))

    for (expected <- expectedCenters) {
      val closest = centroids.minBy(c => StandardMetric.Euclidean.distance(c, expected))
      assert(StandardMetric.Euclidean.distance(closest, expected) < 2.0f,
        s"No centroid found near expected center (${expected.mkString(",")})")
    }
  }

  test("KMeans.fit: handles edge case where numSamples <= k") {
    val data = Array(Array(1.0f, 2.0f), Array(3.0f, 4.0f))
    val centroids = LocalTraining.localKMeans(data, k = 5, maxIterations = 20)
    assert(centroids.length === 5)
  }
}
