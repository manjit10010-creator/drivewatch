# DriveWatch Spark image: pinned Python + Java + PySpark so results are repeatable.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

RUN pip install --no-cache-dir pyspark==3.5.3 duckdb pandas pyarrow

WORKDIR /workspace
