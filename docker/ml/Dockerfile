FROM drivewatch-spark
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir scikit-learn==1.9.1 lightgbm==4.7.0 shap==0.51.0 mlflow==3.18.0 matplotlib
