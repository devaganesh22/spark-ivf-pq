import knn.SparkIVFPQ
import knn.IVFPQCodebook
import org.apache.spark.sql.SparkSession

// 1. Initialize Spark (if not already provided by spark-shell)
val spark = SparkSession.builder()
  .appName("IVF-PQ GloVe Train Example")
  .master("local[*]")
  .getOrCreate()

// 2. Define Parameters for GloVe-100
val basePath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/glove-100_base.parquet"
val codebookPath = "C:/workspace/spark-ivf-pq/ann-sample/glove-100/codebook.bin"

println(s"Loading GloVe-100 database vectors from $basePath...")
val baseDf = spark.read.parquet(basePath)

// 3. Configure the IVF-PQ Estimator for Cosine Distance
val estimator = new SparkIVFPQ()
  .setNumCoarseCentroids(4096)
  .setNumSubspaces(20) // 100 dimensions / 20 = 5 dim per subspace
  .setMaxIterations(20)
  .setMetricName("cosine") // GloVe uses angular/cosine distance

// 4. Train the Model
println(s"Training IVF-PQ Model on GloVe-100 dataset...")
val model = estimator.fit(baseDf, "vector")

// 5. Save the compiled Codebook
println(s"Saving trained codebook to $codebookPath...")
IVFPQCodebook.save(model.codebook, codebookPath)
println("GloVe-100 Training completed successfully!")

sys.exit(0)
