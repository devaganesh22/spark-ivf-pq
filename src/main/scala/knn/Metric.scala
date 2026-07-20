package knn

import scala.math.sqrt

trait Metric[A] extends Serializable {
  def distance(x: A, y: A): Float
}

object StandardMetric {
  val Euclidean: Metric[Array[Float]] = new Metric[Array[Float]] {
    override def distance(x: Array[Float], y: Array[Float]): Float = {
      var sum = 0.0f
      var i = 0
      val len = x.length
      while (i < len) {
        val diff = x(i) - y(i)
        sum += diff * diff
        i += 1
      }
      sqrt(sum).toFloat
    }
  }

  val Cosine: Metric[Array[Float]] = new Metric[Array[Float]] {
    override def distance(x: Array[Float], y: Array[Float]): Float = {
      var dotProduct = 0.0f
      var normA = 0.0f
      var normB = 0.0f
      var i = 0
      val len = x.length
      while (i < len) {
        val xi = x(i)
        val yi = y(i)
        dotProduct += xi * yi
        normA += xi * xi
        normB += yi * yi
        i += 1
      }
      if (normA == 0.0f || normB == 0.0f) 1.0f
      else 1.0f - (dotProduct / (sqrt(normA) * sqrt(normB)).toFloat)
    }
  }
}

object Metric {
  val isSimdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.FloatVector")
    true
  } catch {
    case _: Throwable => false
  }

  private def loadSimdMetric(fieldName: String): Metric[Array[Float]] = {
    try {
      val clazz = Class.forName("knn.SimdMetricImpl$")
      val module = clazz.getField("MODULE$").get(null)
      val field = clazz.getMethod(fieldName)
      field.invoke(module).asInstanceOf[Metric[Array[Float]]]
    } catch {
      case _: Throwable => null
    }
  }

  lazy val Euclidean: Metric[Array[Float]] = {
    val simd = if (isSimdAvailable) loadSimdMetric("Euclidean") else null
    if (simd != null) simd else StandardMetric.Euclidean
  }

  lazy val Cosine: Metric[Array[Float]] = {
    val simd = if (isSimdAvailable) loadSimdMetric("Cosine") else null
    if (simd != null) simd else StandardMetric.Cosine
  }
}
