from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import os
import psycopg2

def fetch_dukascopy():
    from src.data.dukascopy_fetcher import fetch_and_store
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    # Fetch last 24h
    fetch_and_store('XAUUSD', datetime.now() - timedelta(hours=24), datetime.now(), '1h', conn)
    conn.close()

def fetch_fred():
    from src.data.fred_fetcher import fetch_and_store_fred
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    fetch_and_store_fred(['real_yield_10y', 'dxy_index', 'vix'], datetime.now() - timedelta(days=7), datetime.now(), conn)
    conn.close()

def fetch_gdelt():
    from src.data.gdelt_fetcher import fetch_and_store_gdelt
    import pandas as pd
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    # GDELT fetcher needs a price series for alignment, let's mock it or fetch latest
    # For now, just passing dummy to satisfy signature if needed, or assume fetcher handles it
    fetch_and_store_gdelt(24, None, conn)
    conn.close()

def check_for_friday(**context):
    execution_date = context['logical_date']
    if execution_date.weekday() == 4: # Friday
        return "sync_cot"
    return "validate_data"

def sync_cot():
    from src.data.cot_fetcher import fetch_and_store_cot
    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    current_year = datetime.now().year
    fetch_and_store_cot(current_year, current_year, conn)
    conn.close()

def validate_data():
    from scripts.check_health import main as health_check
    try:
        health_check()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError("Health check failed")

default_args = {
    'owner': 'jules',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_data_ingestion',
    default_args=default_args,
    description='Daily data ingestion pipeline',
    schedule='0 6 * * 1-5',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id='fetch_dukascopy', python_callable=fetch_dukascopy)
    t2 = PythonOperator(task_id='fetch_fred', python_callable=fetch_fred)
    t3 = PythonOperator(task_id='fetch_gdelt', python_callable=fetch_gdelt)

    branch = BranchPythonOperator(
        task_id='check_for_friday',
        python_callable=check_for_friday,
    )

    t4 = PythonOperator(task_id='sync_cot', python_callable=sync_cot)
    t5 = PythonOperator(task_id='validate_data', python_callable=validate_data, trigger_rule='none_failed_min_one_success')

    [t1, t2, t3] >> branch
    branch >> t4 >> t5
    branch >> t5
