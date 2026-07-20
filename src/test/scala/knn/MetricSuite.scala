package knn

import org.scalatest.funsuite.AnyFunSuite

class MetricSuite extends AnyFunSuite {

  val tolerance = 1e-4f

  test("StandardMetric.Euclidean: distance between identical vectors is 0") {
    val v = Array(1.0f, 2.0f, 3.0f)
    assert(StandardMetric.Euclidean.distance(v, v) === 0.0f)
  }

  test("StandardMetric.Euclidean: known distance") {
    val a = Array(0.0f, 0.0f, 0.0f)
    val b = Array(3.0f, 4.0f, 0.0f)
    val dist = StandardMetric.Euclidean.distance(a, b)
    assert(math.abs(dist - 5.0f) < tolerance)
  }

  test("StandardMetric.Euclidean: symmetry") {
    val a = Array(1.0f, 5.0f, -3.0f)
    val b = Array(4.0f, -2.0f, 1.0f)
    assert(math.abs(StandardMetric.Euclidean.distance(a, b) - StandardMetric.Euclidean.distance(b, a)) < tolerance)
  }

  test("StandardMetric.Cosine: distance between identical vectors is 0") {
    val v = Array(1.0f, 2.0f, 3.0f)
    val dist = StandardMetric.Cosine.distance(v, v)
    assert(math.abs(dist) < tolerance)
  }

  test("StandardMetric.Cosine: orthogonal vectors have distance 1") {
    val a = Array(1.0f, 0.0f)
    val b = Array(0.0f, 1.0f)
    val dist = StandardMetric.Cosine.distance(a, b)
    assert(math.abs(dist - 1.0f) < tolerance)
  }

  test("StandardMetric.Cosine: opposite vectors have distance 2") {
    val a = Array(1.0f, 0.0f)
    val b = Array(-1.0f, 0.0f)
    val dist = StandardMetric.Cosine.distance(a, b)
    assert(math.abs(dist - 2.0f) < tolerance)
  }

  test("Metric.Euclidean resolves without crashing (fallback or SIMD)") {
    val a = Array(1.0f, 2.0f, 3.0f)
    val b = Array(4.0f, 5.0f, 6.0f)
    val dist = Metric.Euclidean.distance(a, b)
    val expected = StandardMetric.Euclidean.distance(a, b)
    assert(math.abs(dist - expected) < tolerance)
  }

  test("Metric.Cosine resolves without crashing (fallback or SIMD)") {
    val a = Array(1.0f, 0.0f, 0.0f)
    val b = Array(0.0f, 1.0f, 0.0f)
    val dist = Metric.Cosine.distance(a, b)
    val expected = StandardMetric.Cosine.distance(a, b)
    assert(math.abs(dist - expected) < tolerance)
  }
}
