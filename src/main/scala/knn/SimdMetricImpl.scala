package knn

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.VectorOperators
import scala.math.sqrt

object SimdMetricImpl {
  private val SPECIES: VectorSpecies[java.lang.Float] = FloatVector.SPECIES_PREFERRED.asInstanceOf[VectorSpecies[java.lang.Float]]

  val Euclidean: Metric[Array[Float]] = new Metric[Array[Float]] {
    override def distance(x: Array[Float], y: Array[Float]): Float = {
      val len = x.length
      val bound = SPECIES.loopBound(len)
      var sumVector = FloatVector.zero(SPECIES)
      var i = 0

      while (i < bound) {
        val va = FloatVector.fromArray(SPECIES, x, i)
        val vb = FloatVector.fromArray(SPECIES, y, i)
        val diff = va.sub(vb)
        sumVector = sumVector.add(diff.mul(diff))
        i += SPECIES.length()
      }

      var sum = sumVector.reduceLanes(VectorOperators.ADD)

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
      val len = x.length
      val bound = SPECIES.loopBound(len)
      var dotVector = FloatVector.zero(SPECIES)
      var normAVector = FloatVector.zero(SPECIES)
      var normBVector = FloatVector.zero(SPECIES)
      var i = 0

      while (i < bound) {
        val va = FloatVector.fromArray(SPECIES, x, i)
        val vb = FloatVector.fromArray(SPECIES, y, i)
        dotVector = dotVector.add(va.mul(vb))
        normAVector = normAVector.add(va.mul(va))
        normBVector = normBVector.add(vb.mul(vb))
        i += SPECIES.length()
      }

      var dot = dotVector.reduceLanes(VectorOperators.ADD)
      var normA = normAVector.reduceLanes(VectorOperators.ADD)
      var normB = normBVector.reduceLanes(VectorOperators.ADD)

      while (i < len) {
        val xi = x(i)
        val yi = y(i)
        dot += xi * yi
        normA += xi * xi
        normB += yi * yi
        i += 1
      }

      if (normA == 0.0f || normB == 0.0f) 1.0f
      else 1.0f - (dot / (sqrt(normA) * sqrt(normB)).toFloat)
    }
  }
}
