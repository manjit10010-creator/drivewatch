"""Smoke test: can Spark (inside Docker) read one day of the bronze Parquet?

Run from the repo root, through Docker (see the commands in chat).
Reads ONE day only, so it is light on memory.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.master("local[8]")          # 8 threads, see docs/decisions.md
    .appName("drivewatch-smoke-test")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "16")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Read the file directly (all columns are text in bronze, typing comes later).
path = "data/bronze_parquet/date=2026-06-15/part.parquet"
df = spark.read.parquet(path)

print("Spark version:", spark.version)
print("Rows:", df.count())
print("Columns:", len(df.columns))
print("Failures that day:", df.filter(F.col("failure") == "1").count())
print("Distinct models:", df.select("model").distinct().count())

df.select("date", "serial_number", "model", "failure", "smart_5_raw").show(5, truncate=False)

spark.stop()
print("Smoke test finished OK")
