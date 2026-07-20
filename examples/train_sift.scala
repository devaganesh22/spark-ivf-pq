import knn.SparkIVFPQ
import knn.IVFPQCodebook
import org.apache.spark.sql.SparkSession

// 1. Initialize Spark (if not running inside spark-shell)
val spark = SparkSession.builder()
  .appName("IVF-PQ Train Example")
  .master("local[*]")
  .getOrCreate()

// 2. Define Parameters
val basePath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/sift-128_base.parquet"
val codebookPath = "C:/workspace/spark-ivf-pq/ann-sample/sift-128/codebook.bin"

println(s"Loading database vectors from $basePath...")
val baseDf = spark.read.parquet(basePath)

// 3. Configure the IVF-PQ Estimator (Spark MLlib paradigm)
val estimator = new SparkIVFPQ()
  .setNumCoarseCentroids(4096)
  .setNumSubspaces(16)
  .setMaxIterations(20)
  .setMetricName("euclidean")

// 4. Train the Model
println(s"Training IVF-PQ Model on the cluster...")
val model = estimator.fit(baseDf, "vector")

// 5. Save the compiled Codebook to disk for inference
println(s"Saving trained codebook to $codebookPath...")
IVFPQCodebook.save(model.codebook, codebookPath)
println("Training completed successfully!")

spark.stop()
