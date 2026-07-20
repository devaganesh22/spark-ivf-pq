name := "spark-ivf-pq"
version := "0.1.0"
scalaVersion := "2.12.18"

// Compiler flags for Java Vector API (incubator)
javacOptions ++= Seq(
  "-source", "17",
  "-target", "17",
  "--add-modules", "jdk.incubator.vector"
)

// Scala compiler flags
scalacOptions ++= Seq(
  "-deprecation",
  "-feature",
  "-unchecked"
)

// Dependencies
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql"   % "3.5.9",
  "org.apache.spark" %% "spark-mllib" % "3.5.9",
  "com.github.scopt" %% "scopt"       % "4.1.0",
  "org.scalatest"    %% "scalatest"   % "3.2.18" % Test
)

// Fork the JVM for run and test tasks to support incubator modules and Spark
fork := true
val java17SparkFlags = Seq(
  "--add-modules", "jdk.incubator.vector",
  "--add-opens=java.base/java.lang=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
  "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
  "--add-opens=java.base/java.io=ALL-UNNAMED",
  "--add-opens=java.base/java.net=ALL-UNNAMED",
  "--add-opens=java.base/java.nio=ALL-UNNAMED",
  "--add-opens=java.base/java.util=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
  "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
  "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
  "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
  "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
  "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
)
javaOptions ++= java17SparkFlags

Test / fork := true
Test / javaOptions ++= java17SparkFlags
