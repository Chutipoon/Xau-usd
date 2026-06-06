from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import psycopg2

def run_hmm_predict():
    from src.models.hmm_regime import RegimeHMM
    import pandas as pd

    df = pd.read_parquet("data/feature_matrix.parquet")
    # Columns for HMM: [returns, log_volume, realized_vol, yield_spread, cot_net_long, event_spike_zscore]
    hmm_features = ['returns', 'log_volume', 'realized_vol', 'yield_spread', 'cot_net_long', 'event_spike_zscore']

    model = RegimeHMM.load("models/hmm.pkl")
    # Get latest posterior
    proba = model.predict_proba(df[hmm_features])
    latest_proba = proba[-1]

    return {
        "regime": int(latest_proba.argmax()),
        "posterior": latest_proba.tolist()
    }

def run_lstm_predict():
    from src.models.lstm_signal import LSTMTrainer, FEATURE_COLS
    import pandas as pd
    import numpy as np

    df = pd.read_parquet("data/feature_matrix.parquet")
    trainer = LSTMTrainer.load("models/lstm.pt")

    # Preprocess
    X_raw = df[FEATURE_COLS].values
    X_scaled = trainer.scaler.transform(X_raw)

    # Prepare sequence for the last timestamp
    seq_len = trainer.model.sequence_length
    if len(X_scaled) < seq_len:
        raise ValueError(f"Not enough data for LSTM sequence (need {seq_len}, got {len(X_scaled)})")

    last_seq = X_scaled[-seq_len:].reshape(1, seq_len, -1)
    prediction = trainer.predict(last_seq)

    return float(prediction[0][0])

def run_garch_forecast(**context):
    from src.models.garch_vol import RegimeGARCH
    import pandas as pd
    import numpy as np

    df = pd.read_parquet("data/feature_matrix.parquet")
    hmm_output = context['ti'].xcom_pull(task_ids='run_hmm_predict')
    regime = hmm_output['regime']

    garch = RegimeGARCH()
    returns = df['returns_1h']
    # For simplicity, we use a single regime fit for the current forecast if we don't have history
    garch.fit_all(returns, np.full(len(returns), regime))

    vol_forecast = garch.forecast_vol(regime, frequency='H1')
    pos_size = garch.position_size(regime, frequency='H1')

    return {
        "vol_forecast": vol_forecast,
        "position_size": pos_size
    }

def run_regime_bridge(**context):
    from src.execution.regime_signal_bridge import RegimeSignalBridge
    import numpy as np

    ti = context['ti']
    hmm_output = ti.xcom_pull(task_ids='run_hmm_predict')
    lstm_signal = ti.xcom_pull(task_ids='run_lstm_predict')
    garch_output = ti.xcom_pull(task_ids='run_garch_forecast')

    bridge = RegimeSignalBridge()
    forecast = bridge.translate(
        hmm_posterior=np.array(hmm_output['posterior']),
        lstm_signal=lstm_signal,
        garch_position_size=garch_output['position_size']
    )

    return forecast

def store_signals(**context):
    import json
    import os
    import psycopg2

    ti = context['ti']
    hmm_output = ti.xcom_pull(task_ids='run_hmm_predict')
    lstm_signal = ti.xcom_pull(task_ids='run_lstm_predict')
    garch_output = ti.xcom_pull(task_ids='run_garch_forecast')
    bridge_forecast = ti.xcom_pull(task_ids='run_regime_bridge')

    db_url = os.getenv('TIMESCALE_URL', 'postgresql://localhost/xauusd')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO signals (timestamp, hmm_regime, hmm_posterior, lstm_signal, garch_vol, bridge_forecast)
        VALUES (NOW(), %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp) DO UPDATE SET
            hmm_regime = EXCLUDED.hmm_regime,
            hmm_posterior = EXCLUDED.hmm_posterior,
            lstm_signal = EXCLUDED.lstm_signal,
            garch_vol = EXCLUDED.garch_vol,
            bridge_forecast = EXCLUDED.bridge_forecast
    """, (
        hmm_output['regime'],
        json.dumps(hmm_output['posterior']),
        lstm_signal,
        garch_output['vol_forecast'],
        bridge_forecast
    ))

    conn.commit()
    cur.close()
    conn.close()
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
