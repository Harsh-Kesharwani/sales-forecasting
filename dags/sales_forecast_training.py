from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

# Only import essential libraries at the top level
# Move heavy imports (pandas, sklearn, etc.) inside tasks

# Add include path at runtime, not import time
import sys

default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "start_date": datetime(2025, 8, 31),
    "email_on_failure": True,
    "email_on_retry": False,
    "email": ["admin@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    schedule="@weekly",
    start_date=datetime(2025, 7, 22),
    catchup=False,
    default_args=default_args,
    description="Train sales forecasting models",
    tags=["ml", "training", "sales"],
)
def sales_forecast_training():
    
    @task()
    def extract_data_task():
        # Move imports inside the task
        import pandas as pd
        import os
        sys.path.append("/usr/local/airflow/include")
        from utils.data_generator import RealisticSalesDataGenerator

        data_output_dir = "/tmp/sales_data"
        generator = RealisticSalesDataGenerator(
            start_date="2021-01-01", end_date="2021-02-28"  # Fixed invalid date
        )
        print("Generating realistic sales data...")
        file_paths = generator.generate_sales_data(output_dir=data_output_dir)
        total_files = sum(len(paths) for paths in file_paths.values())
        print(f"Generated {total_files} files:")
        for data_type, paths in file_paths.items():
            print(f"  - {data_type}: {len(paths)} files")
        return {
            "data_output_dir": data_output_dir,
            "file_paths": file_paths,
            "total_files": total_files,
        }

    @task()
    def validate_data_task(extract_result):
        # Move imports inside the task
        import pandas as pd
        import glob
        sys.path.append("/usr/local/airflow/include")
        from data_validation.validators import DataValidator

        file_paths = extract_result["file_paths"]
        total_rows = 0
        issues_found = []
        print(f"Validating {len(file_paths['sales'])} sales files...")
        for i, sales_file in enumerate(file_paths["sales"][:10]):
            df = pd.read_parquet(sales_file)
            if i == 0:
                print(f"Sales data columns: {df.columns.tolist()}")
            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue
            required_cols = [
                "date",
                "store_id",
                "product_id",
                "quantity_sold",
                "revenue",
            ]
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")
            total_rows += len(df)
            if df["quantity_sold"].min() < 0:
                issues_found.append(f"Negative quantities in {sales_file}")
            if df["revenue"].min() < 0:
                issues_found.append(f"Negative revenue in {sales_file}")
        
        for data_type in ["promotions", "store_events", "customer_traffic"]:
            if data_type in file_paths and file_paths[data_type]:
                sample_file = file_paths[data_type][0]
                df = pd.read_parquet(sample_file)
                print(f"{data_type} data shape: {df.shape}")
                print(f"{data_type} columns: {df.columns.tolist()}")
        
        validation_summary = {
            "total_files_validated": len(file_paths["sales"][:10]),
            "total_rows": total_rows,
            "issues_found": len(issues_found),
            "issues": issues_found[:5],
        }
        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:5]:
                print(f"  - {issue}")
        else:
            print(f"Validation passed! Total rows: {total_rows}")
        return validation_summary

    @task()
    def train_models_task(extract_result, validation_summary):
        # Move imports inside the task - this is crucial for heavy libraries
        import pandas as pd
        sys.path.append("/usr/local/airflow/include")
        from ml_models.train_models import ModelTrainer
        from utils.mlflow_utils import MLflowManager

        file_paths = extract_result["file_paths"]
        print("Loading sales data from multiple files...")
        sales_dfs = []
        max_files = 50
        for i, sales_file in enumerate(file_paths["sales"][:max_files]):
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} files...")
        
        sales_df = pd.concat(sales_dfs, ignore_index=True)
        print(f"Combined sales data shape: {sales_df.shape}")
        
        daily_sales = (
            sales_df.groupby(["date", "store_id", "product_id", "category"])
            .agg(
                {
                    "quantity_sold": "sum",
                    "revenue": "sum",
                    "cost": "sum",
                    "profit": "sum",
                    "discount_percent": "mean",
                    "unit_price": "mean",
                }
            )
            .reset_index()
        )
        daily_sales = daily_sales.rename(columns={"revenue": "sales"})
        
        # Handle promotions data if available
        if file_paths.get("promotions"):
            promo_df = pd.read_parquet(file_paths["promotions"][0])
            promo_summary = (
                promo_df.groupby(["date", "product_id"])["discount_percent"]
                .max()
                .reset_index()
            )
            promo_summary["has_promotion"] = 1
            daily_sales = daily_sales.merge(
                promo_summary[["date", "product_id", "has_promotion"]],
                on=["date", "product_id"],
                how="left",
            )
            daily_sales["has_promotion"] = daily_sales["has_promotion"].fillna(0)
        
        # Handle customer traffic data if available
        if file_paths.get("customer_traffic"):
            traffic_dfs = []
            for traffic_file in file_paths["customer_traffic"][:10]:
                traffic_dfs.append(pd.read_parquet(traffic_file))
            traffic_df = pd.concat(traffic_dfs, ignore_index=True)
            traffic_summary = (
                traffic_df.groupby(["date", "store_id"])
                .agg({"customer_traffic": "sum", "is_holiday": "max"})
                .reset_index()
            )
            daily_sales = daily_sales.merge(
                traffic_summary, on=["date", "store_id"], how="left"
            )
        
        print(f"Final training data shape: {daily_sales.shape}")
        print(f"Columns: {daily_sales.columns.tolist()}")
        
        # Initialize trainer and prepare data
        trainer = ModelTrainer()
        store_daily_sales = (
            daily_sales.groupby(["date", "store_id"])
            .agg(
                {
                    "sales": "sum",
                    "quantity_sold": "sum",
                    "profit": "sum",
                    "has_promotion": "mean",
                    "customer_traffic": "first",
                    "is_holiday": "first",
                }
            )
            .reset_index()
        )
        store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])
        
        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales,
            target_col="sales",
            date_col="date",
            group_cols=["store_id"],
            categorical_cols=["store_id"],
        )
        print(
            f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}"
        )
        
        # Train models
        results = trainer.train_all_models(
            train_df, val_df, test_df, target_col="sales", use_optuna=True
        )
        
        for model_name, model_results in results.items():
            if "metrics" in model_results:
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f"  {metric}: {value:.4f}")
        
        print("\nVisualization charts have been generated and saved to MLflow/MinIO")
        print("Charts include:")
        print("  - Model metrics comparison")
        print("  - Predictions vs actual values")
        print("  - Residuals analysis")
        print("  - Error distribution")
        print("  - Feature importance comparison")
        
        # Serialize results for task communication
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results.get("metrics", {})
            }
        
        # Get MLflow run ID if available
        try:
            import mlflow
            current_run_id = (
                mlflow.active_run().info.run_id if mlflow.active_run() else None
            )
        except ImportError:
            current_run_id = None
        
        return {
            "training_results": serializable_results,
            "mlflow_run_id": current_run_id,
        }

    @task()
    def evaluate_models_task(training_result):
        # Move imports inside the task
        sys.path.append("/usr/local/airflow/include")
        from utils.mlflow_utils import MLflowManager

        results = training_result["training_results"]
        mlflow_manager = MLflowManager()
        
        best_model_name = None
        best_rmse = float("inf")
        
        for model_name, model_results in results.items():
            if "metrics" in model_results and "rmse" in model_results["metrics"]:
                if model_results["metrics"]["rmse"] < best_rmse:
                    best_rmse = model_results["metrics"]["rmse"]
                    best_model_name = model_name
        
        print(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        try:
            best_run = mlflow_manager.get_best_model(metric="rmse", ascending=True)
            run_id = best_run["run_id"]
        except Exception as e:
            print(f"Could not get best run from MLflow: {e}")
            run_id = training_result.get("mlflow_run_id", "unknown")
        
        return {"best_model": best_model_name, "best_run_id": run_id}

    @task()
    def register_best_model_task(evaluation_result):
        # Move imports inside the task
        sys.path.append("/usr/local/airflow/include")
        from utils.mlflow_utils import MLflowManager

        best_model = evaluation_result["best_model"]
        run_id = evaluation_result["best_run_id"]
        mlflow_manager = MLflowManager()
        
        model_versions = {}
        for model_name in ["xgboost", "lightgbm"]:
            try:
                version = mlflow_manager.register_model(run_id, model_name, model_name)
                model_versions[model_name] = version
                print(f"Registered {model_name} version: {version}")
            except Exception as e:
                print(f"Failed to register {model_name}: {e}")
                model_versions[model_name] = "failed"
        
        return model_versions

    @task()
    def transition_to_production_task(model_versions):
        # Move imports inside the task
        sys.path.append("/usr/local/airflow/include")
        from utils.mlflow_utils import MLflowManager

        mlflow_manager = MLflowManager()
        
        for model_name, version in model_versions.items():
            if version != "failed":
                try:
                    mlflow_manager.transition_model_stage(model_name, version, "Production")
                    print(f"Transitioned {model_name} v{version} to Production")
                except Exception as e:
                    print(f"Failed to transition {model_name} to production: {e}")
        
        return "Models transitioned to production"

    @task()
    def generate_performance_report_task(training_result, validation_summary):
        import json
        
        results = training_result["training_results"]
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_rows": (
                    validation_summary.get("total_rows", 0) if validation_summary else 0
                ),
                "files_validated": (
                    validation_summary.get("total_files_validated", 0)
                    if validation_summary
                    else 0
                ),
                "issues_found": (
                    validation_summary.get("issues_found", 0)
                    if validation_summary
                    else 0
                ),
                "issues": (
                    validation_summary.get("issues", []) if validation_summary else []
                ),
            },
            "model_performance": {},
        }
        
        if results:
            for model_name, model_results in results.items():
                if "metrics" in model_results:
                    report["model_performance"][model_name] = model_results["metrics"]
        
        with open("/tmp/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("Performance report generated")
        print(f"Models trained: {list(report['model_performance'].keys())}")
        return report

    # Define task dependencies
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = train_models_task(extract_result, validation_summary)
    evaluation_result = evaluate_models_task(training_result)
    model_versions = register_best_model_task(evaluation_result)
    transition = transition_to_production_task(model_versions)
    report = generate_performance_report_task(training_result, validation_summary)
    
    # Cleanup task
    cleanup = BashOperator(
        task_id="cleanup",
        bash_command="rm -rf /tmp/sales_data /tmp/performance_report.json || true",
    )
    
    # Set final dependency
    report >> cleanup


# Create the DAG instance
sales_forecast_training_dag = sales_forecast_training()