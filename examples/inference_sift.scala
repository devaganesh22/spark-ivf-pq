import knn.SparkIVFPQModel
import knn.IVFPQCodebook
import org.apache.spark.sql.SparkSession

// 1. Initialize Spark (if not running inside spark-shell)
val spark = SparkSession.builder()
  .appName("IVF-PQ Inference Example")
  .master("local[*]")
  .getOrCreate()

// 2. Define Parameters
val codebookPath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/codebook.bin"
val basePath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/sift-128_base.parquet"
val queryPath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/sift-128_query.parquet"
val outputPath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/query_ivfpq.parquet"

println("Loading Codebook, Database, and Queries...")
val codebook = IVFPQCodebook.load(codebookPath)
val baseDf = spark.read.parquet(basePath)
val queryDf = spark.read.parquet(queryPath)

// 3. Configure the IVF-PQ Transformer (Spark MLlib paradigm)
val model = new SparkIVFPQModel(codebook)
  .setKApprox(10000)   // Stage 1: Fast Approximate Search (Wide Net)
  .setNprobe(256)      // Stage 1: Coarse Centroids to visit
  .setK(100)           // Stage 2: Exact Euclidean Re-Ranking limit

// 4. Transform (Run the Two-Stage Search)
println(s"Running Distributed Two-Stage Search Pipeline...")
val searchStart = System.currentTimeMillis()

// This one line triggers the distributed Encoding, Broadcast, Approximate Search, and Exact Shuffle Join
val resultsDf = model.transform(queryDf, baseDf, "vector")

// Trigger execution and calculate latency
val numQueries = resultsDf.count()
val searchEnd = System.currentTimeMillis()

val latencyMs = (searchEnd - searchStart).toDouble / numQueries
println(f"Search completed in ${(searchEnd - searchStart) / 1000.0} seconds.")
println(f"Average Latency: $latencyMs%.2f ms / query")

// 5. Save results
println(s"Saving final results to $outputPath...")
resultsDf.write.mode("overwrite").parquet(outputPath)

spark.stop()
