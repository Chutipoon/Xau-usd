import pytest
from airflow.models import DagBag
from airflow.utils.db import initdb
import os

@pytest.fixture(scope="session", autouse=True)
def setup_airflow_db():
    """Initialize Airflow database for tests"""
    # Use a temporary sqlite DB for tests to avoid conflicts
    os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = 'sqlite:////tmp/airflow_test.db'
    initdb()
    yield
    if os.path.exists('/tmp/airflow_test.db'):
        try:
            os.remove('/tmp/airflow_test.db')
        except:
            pass

def test_dag_loading():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
    assert 'dag_data_ingestion' in dagbag.dags
    assert 'dag_feature_pipeline' in dagbag.dags
    assert 'dag_signal_generation' in dagbag.dags

def test_dag_data_ingestion_dependencies():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('dag_data_ingestion')

    # Check dependencies
    # [t1, t2, t3] >> branch
    # branch >> t4 >> t5
    # branch >> t5

    branch = dag.get_task('check_for_friday')
    t1 = dag.get_task('fetch_dukascopy')
    t2 = dag.get_task('fetch_fred')
    t3 = dag.get_task('fetch_gdelt')
    t4 = dag.get_task('sync_cot')
    t5 = dag.get_task('validate_data')

    assert branch.upstream_task_ids == {'fetch_dukascopy', 'fetch_fred', 'fetch_gdelt'}
    assert t4.upstream_task_ids == {'check_for_friday'}
    assert t5.upstream_task_ids == {'sync_cot', 'check_for_friday'}

def test_dag_feature_pipeline_dependencies():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('dag_feature_pipeline')

    # [t1, t2] >> t3
    t1 = dag.get_task('compute_gdelt_features')
    t2 = dag.get_task('compute_technical_features')
    t3 = dag.get_task('assemble_feature_matrix')

    assert t3.upstream_task_ids == {'compute_gdelt_features', 'compute_technical_features'}

def test_dag_signal_generation_dependencies():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('dag_signal_generation')

    # [t1, t2, t3] >> t4 >> t5
    t1 = dag.get_task('run_hmm_predict')
    t2 = dag.get_task('run_lstm_predict')
    t3 = dag.get_task('run_garch_forecast')
    t4 = dag.get_task('run_regime_bridge')
    t5 = dag.get_task('store_signals')

    assert t4.upstream_task_ids == {'run_hmm_predict', 'run_lstm_predict', 'run_garch_forecast'}
    assert t5.upstream_task_ids == {'run_regime_bridge'}
