from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta

def fetch_dukascopy():
    print("Fetching Dukascopy OHLCV data...")
    return "dukascopy_data_ref"

def fetch_fred():
    print("Fetching FRED series...")
    return "fred_data_ref"

def fetch_gdelt():
    print("Fetching last 24h GDELT documents...")
    return "gdelt_data_ref"

def check_for_friday(**context):
    execution_date = context['logical_date']
    if execution_date.weekday() == 4: # Friday
        return "sync_cot"
    return "validate_data"

def sync_cot():
    print("Syncing COT data...")
    return "cot_data_ref"

def validate_data():
    print("Validating all tables have today's data...")
    return True

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
