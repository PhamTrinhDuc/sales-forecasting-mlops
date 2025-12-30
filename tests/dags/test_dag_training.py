import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from airflow.models import DagBag


class TestSalesForecastTrainingDAG:
    """Test suite for sales_forecast_training DAG"""
    
    @pytest.fixture
    def dagbag(self): 
        """Load DAG from dagbag"""
        return DagBag(dag_folder='/usr/local/airflow/dags', include_examples=False)
    
    def test_dag_loads(self, dagbag):
        """Test that DAG loads without errors"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        assert dag is not None
        assert dag.dag_id == 'sales_forecast_training'
    
    def test_dag_has_tasks(self, dagbag):
        """Test that DAG has expected tasks"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        task_ids = [task.task_id for task in dag.tasks]
        
        expected_tasks = [
            'extract_data_task',
            'validate_data_task',
            'transform_data_task',
            'train_model_task'
        ]
        
        for task_id in expected_tasks:
            assert task_id in task_ids, f"Task {task_id} not found in DAG"
    
    def test_dag_task_dependencies(self, dagbag):
        """Test that task dependencies are correct"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        
        # Get tasks
        extract_task = dag.get_task('extract_data_task')
        validate_task = dag.get_task('validate_data_task')
        transform_task = dag.get_task('transform_data_task')
        train_task = dag.get_task('train_model_task')
        
        # Check dependencies
        assert extract_task.downstream_list == [validate_task, transform_task]
        assert validate_task.upstream_list == [extract_task]
        assert transform_task.upstream_list == [extract_task]
        assert train_task.upstream_list == [transform_task]
    
    def test_dag_schedule(self, dagbag):
        """Test that DAG has correct schedule"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        assert dag.schedule_interval == '@weekly'
        assert dag.catchup is False
    
    def test_dag_tags(self, dagbag):
        """Test that DAG has correct tags"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        expected_tags = ["sales_forecasting", "training"]
        assert set(dag.tags) == set(expected_tags)
    
    def test_dag_owner(self, dagbag):
        """Test that DAG has correct owner"""
        dag = dagbag.get_dag(dag_id='sales_forecast_training')
        assert dag.owner == "Jiyuu"
    
    def test_extract_data_task_output(self):
        """Test extract_data_task output structure"""
        from dags.sales_forecast_training import sales_forecast_training
        
        # Create mock S3Manager
        with patch('dags.sales_forecast_training.s3_manager') as mock_s3:
            mock_s3.validate_object_in_bucket.return_value = False
            mock_s3.get_files.return_value = {}
            
            with patch('dags.sales_forecast_training.RealisticSalesDataGenerator') as mock_gen:
                mock_gen_instance = Mock()
                mock_gen_instance.generate_sales_data.return_value = {
                    'sales': [{'key': 'sales/data1.csv', 'url': 'http://localhost:9000/...'}],
                    'inventory': [],
                    'customer_traffic': [],
                    'promotions': [],
                    'store_events': []
                }
                mock_gen.return_value = mock_gen_instance
                
                # Verify output structure
                assert 'file_paths' in ['file_paths', 'total_files', 'data_output_dir']
                assert 'total_files' in ['file_paths', 'total_files', 'data_output_dir']


class TestDAGTaskLogic:
    """Test logic of individual tasks"""
    
    def test_validate_data_structure(self):
        """Test validate_data_task returns correct structure"""
        validate_summary = {
            "total_files_checked": 10,
            "total_rows": 1000,
            "total_issues": 0,
            "issues_found": []
        }
        
        assert 'total_files_checked' in validate_summary
        assert 'total_rows' in validate_summary
        assert 'total_issues' in validate_summary
        assert 'issues_found' in validate_summary
    
    def test_transform_data_output_structure(self):
        """Test transform_data_task output is DataFrame"""
        # Create sample DataFrame
        daily_store_sales = pd.DataFrame({
            'date': pd.date_range('2025-01-01', periods=3),
            'store_id': [1, 1, 2],
            'quantity_sold': [100, 150, 200],
            'sales': [5000, 7500, 10000],
            'profit': [1000, 1500, 2000],
            'has_promotion': [0.5, 0.6, 0.4],
            'customer_traffic': [500, 600, 700],
            'is_holiday': [0, 0, 1]
        })
        
        assert isinstance(daily_store_sales, pd.DataFrame)
        assert len(daily_store_sales) > 0
        assert 'date' in daily_store_sales.columns
        assert 'store_id' in daily_store_sales.columns
        assert 'sales' in daily_store_sales.columns
