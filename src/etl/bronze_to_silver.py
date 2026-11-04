"""Bronze -> silver: type the columns, quarantine bad rows, drop duplicate drive-days.

Run from the repo root, through Docker:
  docker run --rm -v "${PWD}:/workspace" drivewatch-spark \
      python src/etl/bronze_to_silver.py --start 2026-06-01 --end 2026-06-30

Reads  data/bronze_parquet/date=YYYY-MM-DD/part.parquet   (all columns are text)
Writes data/silver/year=YYYY/month=M/                     (typed, de-duplicated)
       data/quarantine/run=<start>_<end>/                 (rejected rows + reason)
"""
import argparse
import os
from datetime import date, timedelta

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BRONZE = "data/bronze_parquet"
SILVER = "data/silver"
QUARANTINE = "data/quarantine"

TEXT_COLS = {"serial_number", "model", "datacenter"}
INT_COLS = {"failure", "cluster_id", "vault_id", "pod_id", "pod_slot_num"}


def day_paths(start, end):
    """One bronze file per day; skip days that do not exist."""
    paths, d = [], start
    while d <= end:
        p = f"{BRONZE}/date={d.isoformat()}/part.parquet"
        if os.path.exists(p):
            paths.append(p)
        d += timedelta(days=1)
    return paths


def typed_columns(columns):
    """Build one select list that gives every column its real type.

    Casting text that is not a valid number gives null (Spark default), and
    impossible values (negative counts, normalized outside 0-255) are nulled.
    """
    exprs = []
    for c in columns:
        col = F.col(c)
        if c == "date":
            exprs.append(F.to_date(col).alias(c))
        elif c in TEXT_COLS:
            exprs.append(F.trim(col).alias(c))
        elif c in INT_COLS:
            exprs.append(col.cast("int").alias(c))
        elif c == "capacity_bytes":
            exprs.append(col.cast("long").alias(c))
        elif c == "is_legacy_format":
            exprs.append(col.cast("boolean").alias(c))
        elif c.endswith("_raw"):
            v = col.cast("long")
            exprs.append(F.when(v >= 0, v).alias(c))
        elif c.endswith("_normalized"):
            v = col.cast("int")
            exprs.append(F.when(v.between(0, 255), v).alias(c))
        else:
            exprs.append(col)
    # keep the original text of two key columns so quarantined rows show what arrived
    exprs.append(F.col("date").alias("date_text"))
    exprs.append(F.col("failure").alias("failure_text"))
    return exprs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-06-30")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    spark = (
        SparkSession.builder.master("local[8]")
        .appName("drivewatch-bronze-to-silver")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    paths = day_paths(start, end)
    if not paths:
        raise SystemExit(f"No bronze files found between {start} and {end}")
    print(f"Reading {len(paths)} day files ({start} to {end})")

    raw = spark.read.option("mergeSchema", "true").parquet(*paths)
    print("Input rows:", raw.count())

    typed = raw.select(*typed_columns(raw.columns))

    # Reason for rejection, first matching rule wins; null means the row is fine.
    typed = typed.withColumn(
        "reason",
        F.when(F.col("date").isNull(), "bad_date")
        .when(F.col("serial_number").isNull() | (F.col("serial_number") == ""), "missing_serial")
        .when(F.col("model").isNull() | (F.col("model") == ""), "missing_model")
        .when(F.col("failure").isNull() | ~F.col("failure").isin(0, 1), "bad_failure_flag")
        .when(F.col("capacity_bytes").isNull() | (F.col("capacity_bytes") <= 0), "bad_capacity"),
    )

    q_cols = ["date_text", "failure_text", "serial_number", "model", "capacity_bytes", "reason"]
    bad = typed.filter(F.col("reason").isNotNull())
    good = typed.filter(F.col("reason").isNull()).drop("reason")
    quarantine = bad.select(*q_cols)

    # Duplicate (date, serial_number) pairs: cheap check on two columns first.
    dup_keys = good.groupBy("date", "serial_number").count().filter(F.col("count") > 1)
    n_dup_keys = dup_keys.count()
    print("Duplicate drive-day keys:", n_dup_keys)
    if n_dup_keys > 0:
        w = Window.partitionBy("date", "serial_number").orderBy(F.col("failure").desc())
        ranked = good.withColumn("rn", F.row_number().over(w))
        dupes = (
            ranked.filter("rn > 1")
            .withColumn("reason", F.lit("duplicate_drive_day"))
            .select(*q_cols)
        )
        quarantine = quarantine.unionByName(dupes)
        good = ranked.filter("rn = 1").drop("rn")

    silver = (
        good.drop("date_text", "failure_text")
        .withColumn("year", F.year("date"))
        .withColumn("month", F.month("date"))
    )

    run_tag = f"{start}_{end}"
    silver.write.mode("overwrite").partitionBy("year", "month").parquet(SILVER)
    quarantine.write.mode("overwrite").parquet(f"{QUARANTINE}/run={run_tag}")

    # Report from what was actually written.
    silver_n = (
        spark.read.parquet(SILVER)
        .filter(F.col("date").between(str(start), str(end)))
        .count()
    )
    print("Silver rows written:", silver_n)
    try:
        q = spark.read.parquet(f"{QUARANTINE}/run={run_tag}")
        print("Quarantined rows by reason:")
        q.groupBy("reason").count().show()
    except Exception:
        print("Quarantined rows: 0")

    spark.stop()
    print("bronze_to_silver finished OK")


if __name__ == "__main__":
    main()
