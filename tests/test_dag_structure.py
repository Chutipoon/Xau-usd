import pytest
from airflow.models import DagBag
import os

@pytest.fixture(scope="session", autouse=True)
def setup_airflow_db():
    """Initialize Airflow database for tests"""
    os.environ['AIRFLOW__DATABASE__SQL_ALCHEMY_CONN'] = 'sqlite:////tmp/airflow_test.db'
    from airflow.utils import db
    db.resetdb()
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
    branch = dag.get_task('check_for_friday')
    assert branch.upstream_task_ids == {'fetch_dukascopy', 'fetch_fred', 'fetch_gdelt'}

def test_dag_feature_pipeline_dependencies():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('dag_feature_pipeline')
    t3 = dag.get_task('assemble_feature_matrix')
    assert t3.upstream_task_ids == {'compute_gdelt_features', 'compute_technical_features'}

def test_dag_signal_generation_dependencies():
    dagbag = DagBag(dag_folder='dags/', include_examples=False)
    dag = dagbag.get_dag('dag_signal_generation')
    t4 = dag.get_task('run_regime_bridge')
    assert t4.upstream_task_ids == {'run_hmm_predict', 'run_lstm_predict', 'run_garch_forecast'}
