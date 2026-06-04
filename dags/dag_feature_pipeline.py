from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import psycopg2
import pandas as pd

def compute_gdelt_features():
    from src.data.gdelt_fetcher import compute_gdelt_features
    # This usually requires latest prices from DB
    print("Computing GDELT features...")
    return "gdelt_features_ref"

def compute_technical_features():
    print("Computing Technical features (RSI, MACD, etc.)...")
    return "tech_features_ref"

def assemble_feature_matrix():
    print("Assembling feature matrix and saving as parquet...")
    return "feature_matrix_path"

default_args = {
    'owner': 'jules',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_feature_pipeline',
    default_args=default_args,
    description='Feature engineering pipeline',
    schedule='30 6 * * 1-5',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id='compute_gdelt_features', python_callable=compute_gdelt_features)
    t2 = PythonOperator(task_id='compute_technical_features', python_callable=compute_technical_features)
    t3 = PythonOperator(task_id='assemble_feature_matrix', python_callable=assemble_feature_matrix)

    [t1, t2] >> t3
