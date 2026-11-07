# DriveWatch API image (development version, step 5c).
# Built on the ML image so DuckDB, LightGBM and MLflow are the SAME versions used for training.
# FastAPI and uvicorn are installed unpinned for now; step 5e pins every version and slims the image.
#   docker build -t drivewatch-api docker/api
#   docker run --rm -p 8000:8000 -v "${PWD}:/workspace" drivewatch-api
FROM drivewatch-ml

WORKDIR /workspace
RUN pip install --no-cache-dir fastapi uvicorn

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]