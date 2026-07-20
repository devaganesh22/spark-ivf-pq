package knn
import org.apache.spark.sql.SparkSession

object ReadGT {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ReadGT").master("local[*]").getOrCreate()
    val gt = spark.read.parquet("C:/workspace/spark-ivf-pq/ann-sample/sift-128/sift-128_groundtruth.parquet")
    
    println("=== GROUND TRUTH SCHEMA ===")
    gt.printSchema()
    
    println("=== FIRST 2 ROWS (Truncated to first 10 elements) ===")
    // We only take the first 10 elements of the arrays so it doesn't flood the console
    val rows = gt.take(2)
    rows.foreach { row =>
      val neighbors = row.getSeq[Int](0).take(10).mkString("[", ", ", ", ...]")
      val distances = row.getSeq[Float](1).take(10).mkString("[", ", ", ", ...]")
      println(s"Neighbors: $neighbors")
      println(s"Distances: $distances")
      println("--------------------------")
    }
    
    spark.stop()
  }
}
