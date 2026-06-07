from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import psycopg2
import pandas as pd

DAG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(DAG_DIR)

def compute_gdelt_features():
    from src.data.gdelt_fetcher import fetch_and_store_gdelt
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)

    # Need latest price for tone_price_divergence
    # Fixed: Use 'ts' and 'close_price' as per schema.sql
    df_price = pd.read_sql("SELECT ts, close_price FROM ohlcv_xauusd ORDER BY ts DESC LIMIT 1000", conn)
    df_price['ts'] = pd.to_datetime(df_price['ts'], utc=True)
    price_series = df_price.set_index('ts')['close_price'].sort_index()

    fetch_and_store_gdelt(hours_back=24, price_series=price_series, db_conn=conn)
    conn.close()
    return "gdelt_features_updated"

def compute_technical_features():
    # Technical features are now calculated within assemble_feature_matrix
    print("Technical indicators will be computed during matrix assembly.")
    return "tech_features_ready"

def assemble_feature_matrix():
    from src.data.feature_engineering import assemble_feature_matrix
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)

    df = assemble_feature_matrix(conn)

    # Use absolute path for Airflow stability
    output_path = os.path.join(PROJECT_ROOT, "data", "feature_matrix.parquet")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path)

    conn.close()
    return output_path

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
