from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_hmm_predict():
    print("Predicting regime posterior with HMM...")
    return "hmm_posterior_data"

def run_lstm_predict():
    print("Predicting direction signal with LSTM...")
    return "lstm_signal_data"

def run_garch_forecast():
    print("Forecasting volatility with GARCH...")
    return "garch_vol_data"

def run_regime_bridge():
    print("Translating outputs to pysystemtrade forecast...")
    return "bridge_forecast_data"

def store_signals():
    print("Storing signals in the database...")
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
    'dag_signal_generation',
    default_args=default_args,
    description='Signal generation pipeline',
    schedule='0 7 * * 1-5',
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(task_id='run_hmm_predict', python_callable=run_hmm_predict)
    t2 = PythonOperator(task_id='run_lstm_predict', python_callable=run_lstm_predict)
    t3 = PythonOperator(task_id='run_garch_forecast', python_callable=run_garch_forecast)
    t4 = PythonOperator(task_id='run_regime_bridge', python_callable=run_regime_bridge)
    t5 = PythonOperator(task_id='store_signals', python_callable=store_signals)

    [t1, t2, t3] >> t4 >> t5
