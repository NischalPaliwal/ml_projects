#type:ignore
from pyspark.sql import SparkSession
from pyspark.sql.functions import lag, avg, stddev, col, datediff, lit
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.window import Window
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from random import randint, random
import numpy as np
import builtins

class InventoryForcaster:
    def __init__(self, app_name="InventoryForecast"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.models = {}
        self.evaluators = {}

    def create_sample_data(self, num_products=100, days=365):
        data = []
        base_date = datetime(2023, 1, 1)

        for product_id in range(1, num_products + 1):
            base_demand = randint(10, 100)
            seasonality_factor = random() * 0.3 + 0.85

            for day in range(days):
                current_date = base_date + timedelta(days=day)
                seasonal_mult = 1 + 0.2 * np.sin(2 * np.pi * day / 365 + np.pi)
                weekly_mult = 1.1 if current_date.weekday() >= 5 else 1.0
                noise = random() * 0.4 + 0.8
                demand = int(base_demand * seasonality_factor * seasonal_mult * weekly_mult * noise)
                stock_level = builtins.max(0, randint(0, 200) - demand)
                price = builtins.round(random() * 50 + 10, 2)

                data.append({
                    'product_id': product_id,
                    'date': current_date,
                    'demand': demand,
                    'stock_level': stock_level,
                    'price': price,
                    'day_of_week': current_date.weekday(),
                    'month': current_date.month,
                    'quarter': (current_date.month - 1) // 3 + 1,
                    'is_weekend': 1 if current_date.weekday() >= 5 else 0
                })

        schema = StructType([
            StructField("product_id", IntegerType(), True),
            StructField("date", DateType(), True),
            StructField("demand", IntegerType(), True),
            StructField("stock_level", IntegerType(), True),
            StructField("price", DoubleType(), True),
            StructField("day_of_week", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("quarter", IntegerType(), True),
            StructField("is_weekend", IntegerType(), True)
        ])

        return self.spark.createDataFrame(data, schema)
    
    def engineer_features(self, df):
        window_7d = Window.partitionBy("product_id").orderBy("date").rowsBetween(-7, -1)
        window_30d = Window.prtitionBy("product_id").orderBy("date").rowsBetween(-30, -1)

        df_features = df.withColumn("demand_lag_1", lag("demand", 1).over(
            Window.partitionBy("product_id").orderBy("date")
        )).withColumn("demand_lag_7", lag("demand", 7).over(
            Window.partitionBy("product_id").orderBy("date")
        )).withColumn("demand_avg_7d", avg("demand").over(window_7d)
        ).withColumn("demand_avg_30d", avg("demand").over(window_30d)
        ).withColumn("demand_std_7d", stddev("demand").over(window_7d)
        ).withColumn("stock_avg_7d", avg("stock_level").over(window_7d)
        ).withColumn("price_change", col("price") - lag("price", 1).over(
            Window.partitionBy("product_id").orderBy("date")
        ))

        df_features = df_features.withColumn("days_since_start", datediff(col("date"), lit("2023-01-01")))

        df_features = df_features.fillna({
            "demand_lag_1": 0,
            "demand_lag_7": 0,
            "demand_avg_7d": 0,
            "demand_avg_30d": 0,
            "demand_std_7d": 0,
            "stock_avg_7d": 0,
            "price_change": 0
        })

        self.df_features = df_features

        return df_features
    
    def prepare_ml_data(self, df, target_col="demand"):
        feature_cols = [
            "product_id", "price", "stock_level", "day_of_week", "month", 
            "quarter", "is_weekend", "demand_lag_1", "demand_lag_7",
            "demand_avg_7d", "demand_avg_30d", "demand_std_7d", 
            "stock_avg_7d", "price_change", "days_since_start"
        ]

        df_clean = df.filter(col(target_col).isNotNull())

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )

        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )

        preprocessing_pipeline = Pipeline(stages=[assembler, scaler])

        return df_clean, preprocessing_pipeline
    
    def train_models(self, df, target_col="demand", test_size=0.2):
        df_clean, preprocessing_pipeline = self.prepare_ml_data(df, target_col)
        train_df, test_df = df_clean.randomSplit([1 - test_size, test_size], seed=42)

        print(f"Training data size: {train_df.count()}")
        print(f"Test data size: {test_df.count()}")

        models_config = {
            'linear_regression': LinearRegression(
                featuresCol="scaled_features",
                labelCol=target_col,
                predictionCol="prediction"
            ),
            'random_forest': RandomForestRegressor(
                featuresCol="scaled_features",
                labelCol=target_col,
                predictionCol="prediction",
                numTrees=50,
                maxDepth=10
            ),
            'gradient_boosting': GBTRegressor(
                featuresCol="scaled_features",
                labelCol=target_col,
                predictionCol="prediction",
                maxIter=50,
                maxDepth=8
            )
        }

        results = {}

        for model_name, model in models_config.items():
            print(f"\nTraining {model_name}...")

            pipeline = Pipeline(stages=preprocessing_pipeline.getStages() + [model])

            model_fitted = pipeline.fit(train_df)

            train_predictions = model_fitted.transform(train_df)
            test_predictions = model_fitted.transform(test_df)

            evaluator = RegressionEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="rmse"
            )

            train_rmse = evaluator.evaluate(train_predictions)
            test_rmse = evaluator.evaluate(test_predictions)

            evaluator_r2 = RegressionEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="r2"
            )

            train_r2 = evaluator_r2.evaluate(train_predictions)
            test_r2 = evaluator_r2.evaluate(test_predictions)

            evaluator_mae = RegressionEvaluator(
                labelCol=target_col,
                predictionCol="prediction",
                metricName="mae"
            )
            
            train_mae = evaluator_mae.evaluate(train_predictions)
            test_mae = evaluator_mae.evaluate(test_predictions)

            results[model_name] = {
                'model': model_fitted,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'predictions': test_predictions
            }

            print(f"Train RMSE: {train_rmse:.2f}")
            print(f"Test RMSE: {test_rmse:.2f}")
            print(f"Test R2: {test_r2:.2f}")
            print(f"Test MAE: {test_mae:.2f}")

        self.models = results
        return results, train_df, test_df
    
    def forecast_inventory(self, model_name, product_id, forecast_days=30):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        model = self.models[model_name]['model']

        latest_data = self.df_features.filter(col("product_id") == product_id) \
                                    .orderBy(desc("date"))  \
                                    .limit(30)
        
        forecasts = []

        for i in range(forecast_days):
            forecast_date = latest_data.first()['date'] + timedelta(days=i+1)

            forecast_row = latest_data.limit(1).withColumn("date", lit(forecast_date))

            prediction = model.transform(forecast_row)
            forecast_value = prediction.select("prediction").first()["prediction"]

            forecasts.append({
                'date': forecast_date,
                'predicted_demand': max(0, int(forecast_value))  # Ensure non-negative
            })

        return forecasts
    
    def print_model_comparison(self):
        if not self.models:
            print("No models trained yet!")
            return
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)

        print(f"{'Model':<20} {'Test RMSE':<12} {'Test R2':<10} {'Test MAE':<10}")
        print("-" * 60)

        for model_name, results in self.models.items():
            print(f"{model_name:<20} {results['test_rmse']:<12.2f} {results['test_r2']:<10.3f} {results['test_mae']:<10.2f}")

    

def main():
    forcaster = InventoryForcaster()

    print("Creating sample inventory data...")
    df = forcaster.create_sample_data(num_products=50, days=300)
    df.show()

if __name__ == "__main__":
    main()