import knn.SparkIVFPQModel
import knn.IVFPQCodebook
import org.apache.spark.sql.SparkSession

// 1. Initialize Spark
val spark = SparkSession.builder()
  .appName("IVF-PQ GloVe Inference Example")
  .master("local[*]")
  .getOrCreate()

// 2. Define Parameters for GloVe-100
val codebookPath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/codebook.bin"
val basePath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/glove-100_base.parquet"
val queryPath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/glove-100_query.parquet"
val outputPath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/query_ivfpq.parquet"

println("Loading GloVe Codebook, Database, and Queries...")
val codebook = IVFPQCodebook.load(codebookPath)
val baseDf = spark.read.parquet(basePath)
val queryDf = spark.read.parquet(queryPath)

// 3. Configure the IVF-PQ Transformer (Autonomous Mode)
val model = new SparkIVFPQModel(codebook)
  .setK(100)           // Only set the exact neighbors desired. Engine handles the rest!
  // .setKApprox(10000) // Optional override: Manual Stage 1 Wide Net
  // .setNprobe(256)    // Optional override: Manual Coarse Centroids to visit

// 4. Transform (Run the Two-Stage Search)
println(s"Running Distributed Two-Stage Search Pipeline using Cosine metric...")
val searchStart = System.currentTimeMillis()

// The transform method abstracts all Spark shuffle and broadcast logic
val resultsDf = model.transform(queryDf, baseDf, "vector")

// Trigger execution
val numQueries = resultsDf.count()
val searchEnd = System.currentTimeMillis()

val latencyMs = (searchEnd - searchStart).toDouble / numQueries
println(f"Search completed in ${(searchEnd - searchStart) / 1000.0} seconds.")
println(f"Average Latency: $latencyMs%.2f ms / query")

// 5. Save results
println(s"Saving final results to $outputPath...")
resultsDf.write.mode("overwrite").parquet(outputPath)

sys.exit(0)
