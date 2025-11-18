"""Sales Analyzer - ETL pipeline"""
from __future__ import annotations
import logging
import warnings
import pyarrow as pa
import json
from pathlib import Path
from typing import Dict, Optional
import duckdb
import pandas as pd
from datetime import date
import great_expectations as ge
from great_expectations.core.batch import Batch
from great_expectations.core.batch_spec import RuntimeDataBatchSpec
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.validator.validator import Validator
from synthetic_data_generator import SyntheticDataGenerator
from build_dimensional_model import DimensionalModelBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("etl_pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

class SalesETL:
    def __init__(self, input_dir: str | Path = "data/input" , output_dir: str | Path = "data/output",
                 warehouse_dir: str | Path = "data/output/sales_analytics.duckdb") -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) 
        self.warehouse_dir = Path(warehouse_dir)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok= True)
        self.synthetic_generator = SyntheticDataGenerator()
        self.df: Optional[pd.DataFrame] = None

    def build_dataset(self, num_synthetic_rows: int = 100_000,
                      start_date = date(2023,1,1),
                      end_date = date(2025,12,31),
                      rebuild: bool = False) -> pd.DataFrame:
        """Create a dataset of reatil sales of company B2B
        Args:
            rebuild: If True, always generate new data. If False, load from file if exists.
        """

        dataset_file = self.output_dir / "synthetic_sales.parquet"

        if not rebuild and dataset_file.exists():
            logger.info("Loading existing dataset from %s", dataset_file)
            self.df = pd.read_parquet(dataset_file)
            logger.info("Loaded %s rows and %s columns", len(self.df), len(self.df.columns))
            return self.df

        logger.info("Generating new synthetic data (rebuild=%s)", rebuild)
        synthetic_df = self.synthetic_generator.generate_synthetic_data(
                num_rows=num_synthetic_rows,start_date = start_date, end_date=end_date )

        synthetic_df = synthetic_df.to_pandas()
        logger.info("Creating a total of  %s  synthetic rows and %s columns ",
                    len(synthetic_df), len(synthetic_df.columns))

        synthetic_df.to_parquet(dataset_file)
        logger.info("Saved dataset to %s", dataset_file)

        self.df = synthetic_df
        return synthetic_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get calculate metrics"""

        if "order_date" in df:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)
            df["order_year"] = df["order_date"].dt.year
            df["order_month"] = df["order_date"].dt.month

        if "price" in df.columns and "quantity" in df.columns and "discount" in df.columns:
            df["revenue"] = df["price"] * df["quantity"] * (1 - df["discount"])
            df["discount_amount"] = df["price"] * df["quantity"] * df["discount"]
            df["gross_sales"] = df["price"] * df["quantity"]
        
        if "ship_latency_days" in df.columns:
            df["is_late_shipment"] = df["ship_latency_days"] > 3

        self.df = df
        logger.debug("df dataset with %s rows and %s columns", len(df), len(df.columns))
        return df

    def build_summaries(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create summary tables for analytics"""
        summaries: Dict[str, pd.DataFrame] = {}

        if "order_year" in df.columns and "revenue" in df.columns:
            summaries["yearly"] = (
                df.groupby("order_year", as_index=False)
                .agg(
                    total_orders=("order_year", "count"),
                    total_revenue=("revenue", lambda x: round(x.sum(), 2)),
                    unique_customers=("customer_id", "nunique"),
                )
                .sort_values("order_year")
            )

        if {"order_year", "segment", "revenue"}.issubset(df.columns):
            summaries["segment_yearly"] = (
                df.groupby(["order_year", "segment"], as_index=False)
                .agg(
                    total_revenue=("revenue", lambda x: round(x.sum(), 2)),
                    total_orders=("segment", "count"),
                )
                .sort_values(["order_year", "total_revenue"], ascending=[True, False])
            )

        if {"region", "revenue"}.issubset(df.columns):
            summaries["regional_revenue"] = (
                df.groupby("region", as_index=False)
                .agg(
                    total_revenue=("revenue", lambda x: round(x.sum(), 2)),
                    total_orders=("region", "count"),
                    unique_customers =("customer_id", "nunique"),
                )
                .sort_values("total_revenue", ascending=False)
            )

        if {"product_name", "revenue"}.issubset(df.columns):
            summaries["top_products"] = (
                df.groupby("product_name", as_index=False)
                .agg(
                    total_revenue=("revenue", lambda x: round(x.sum(), 2)),
                    total_orders=("product_name", "count"),
                )
                .sort_values("total_revenue", ascending=False)
                .head(5)
            )

        logger.info("Built %s summary tables", len(summaries))
        return summaries

    def run_quality_checks(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Run Great Expectations checks on critical columns"""
        pandas_df = df
        context = ge.get_context()
        execution_engine = PandasExecutionEngine()
        execution_engine.data_context = context

        batch_data = execution_engine.get_batch_data( 
            RuntimeDataBatchSpec(batch_data=pandas_df)
        )
        batch = Batch(data=batch_data, data_context=context)
        suite = ExpectationSuite("sales_quality_checks")
        validator = Validator(
            execution_engine=execution_engine,
            data_context=context,
            expectation_suite=suite,
            batches=[batch],
        )
        expectations: Dict[str, bool] = {}

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`result_format` configured at the Validator-level will not be persisted",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="`result_format` configured at the Expectation-level will not be persisted",
                category=UserWarning,
            )

            for column in ["order_id", "customer_id", "price", "order_date"]:
                if column in pandas_df.columns:
                    result = validator.expect_column_values_to_not_be_null(column)
                    expectations[f"{column}_not_null"] = bool(result.success)

            if "price" in pandas_df.columns:
                result = validator.expect_column_values_to_be_between(
                    "price", min_value=0, strict_min=True
                )
                expectations["sales_positive"] = bool(result.success)
                
            if "quantity" in pandas_df.columns:
                result = validator.expect_column_values_to_be_between(
                    "quantity", min_value = 1 , max_value = 100
                )
                expectations["quantity_in_range"] = bool(result.success)

            if "order_year" in pandas_df.columns:
                result = validator.expect_column_values_to_be_between(
                    "order_year",
                    min_value=int(df["order_year"].min()),
                    max_value=int(df["order_year"].max()),
                )
                expectations["order_year_valid_range"] = bool(result.success)

        expectations["overall_success"] = all(expectations.values())
        logger.info("Data quality checks success=%s", expectations["overall_success"])
        return expectations

    def persist_outputs(
        self,
        df: pd.DataFrame,
        summaries: Dict[str, pd.DataFrame],
        quality_results: Dict[str, bool],
    ) -> None:
        """Persist the dataset, summary tables, and quality report under the data/output directory"""
        dataset_path = self.output_dir / "sales_enriched.parquet"
        df.to_parquet(dataset_path, index=False)
        logger.info("Saved enriched dataset to %s", dataset_path)

        for name, table in summaries.items():
            output_path = self.output_dir / f"{name}.csv"
            table.to_csv(output_path, index=False)
            logger.info("Saved %s summary to %s", name, output_path)

        quality_path = self.output_dir / "quality_report.json"
        quality_path.write_text(json.dumps(quality_results, indent=2))
        logger.info("Saved data quality report to %s", quality_path)

    def load_into_warehouse(
        self,
        summaries: Dict[str, pd.DataFrame],
        build_star_schema: bool = True,
    ) -> None:
        """Load summary tables and dimensional model into DuckDB"""

        # Load summary tables
        with duckdb.connect(str(self.warehouse_dir)) as conn:
            for name, table in summaries.items():
                view_name = f"{name}_summary"
                logger.info("Loading %s into warehouse", view_name)
                arrow_summary = pa.Table.from_pandas(table)
                conn.register(view_name, arrow_summary)
                conn.execute(
                    f"CREATE OR REPLACE TABLE analytics.{view_name} AS SELECT * FROM {view_name}"
                )
                conn.unregister(view_name)

        logger.info("Loaded %s summary tables into %s", len(summaries), self.warehouse_dir)

        # Build dimensional model
        if build_star_schema:
            logger.info("Building dimensional model (star schema)")
            try:
                dim_builder = DimensionalModelBuilder(warehouse_path=self.warehouse_dir)
                dim_builder.build_all()
                dim_builder.close()
                logger.info("Dimensional model built successfully")
            except Exception as exc:
                logger.error("Failed to build dimensional model: %s", exc)
                logger.warning("Continuing without dimensional model")

    def run(self, num_synthetic_rows: int = 10_000, build_star_schema: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute the pipeline"""

        combined_df = self.build_dataset(num_synthetic_rows=num_synthetic_rows,rebuild = True)
        transformed_df = self.transform(combined_df)
        summaries = self.build_summaries(transformed_df)
        quality_results = self.run_quality_checks(transformed_df)
        self.persist_outputs(transformed_df, summaries, quality_results)
        #self.load_into_warehouse(summaries, build_star_schema=build_star_schema)

        return transformed_df

if __name__ == "__main__":
    etl = SalesETL()
    try:
        etl.run()
    except ValueError as exc:
        logger.error("Pipeline aborted: %s", exc)
