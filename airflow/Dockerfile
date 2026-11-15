FROM apache/airflow:2.10.5-python3.11

# Same ML library versions as the API image, so the model pickle is compatible
COPY airflow/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /tmp/requirements-airflow.txt