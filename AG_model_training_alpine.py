import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from typing import Dict, Optional
import torch
import logging
import os

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.8)  # Prevent OOM

    # Mixed precision training
    from torch.amp import GradScaler

    scaler = GradScaler("cuda", enabled=True)


# Ensure log directory exists
log_dir = 'C:/Users/USER/Documents/Deepan/Workspaces/Alpine-data-challenge/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'autogluon_training_alpine_valley.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


hyperparameters = {
    # Deep Learning Models
    "DeepAR": {},
    "TemporalFusionTransformer": {},
    "PatchTST": {},
    "TiDE": {},
}


class WeatherForecaster:
    def __init__(self, config: Dict) -> None:
        """
        Initialize the WeatherForecaster with the given configuration.

        Args:
            config (Dict): Configuration dictionary.
        """
        self.config = config
        self.predictor: Optional[TimeSeriesPredictor] = None
        self.full_history: Optional[TimeSeriesDataFrame] = None

    def _load_data(self) -> TimeSeriesDataFrame:
        """
        Load and preprocess time series data.

        Returns:
            TimeSeriesDataFrame: Preprocessed time series data.
        """
        df = pd.read_csv(self.config["data_path"])
        df["timestamp"] = pd.to_datetime(df[self.config["date_column"]])
        df["item_id"] = self.config["valley_ids"][0]
        static_df = pd.DataFrame({"item_id": self.config["valley_ids"]})

        return TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp",
            static_features_df=static_df,
        )

    def _generate_future_covariates(
        self, last_date: pd.Timestamp, prediction_length: int
    ) -> TimeSeriesDataFrame:
        """
        Create future covariates for forecasting.

        Args:
            last_date (pd.Timestamp): The last date in the historical data.
            prediction_length (int): The length of the prediction window.

        Returns:
            TimeSeriesDataFrame: DataFrame containing future covariates.
        """
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1),
            periods=prediction_length,
            freq=self.config["frequency"],
        )

        # Example covariate: day of year sine/cosine
        future_covariates = pd.DataFrame(
            {
                "timestamp": future_dates,
                "day_of_year": future_dates.dayofyear,
                "seasonality": np.sin(2 * np.pi * future_dates.dayofyear / 365),
                "year": future_dates.year,
                "item_id": self.config["valley_ids"][0],
            }
        ).set_index("timestamp")

        return TimeSeriesDataFrame.from_data_frame(
            future_covariates, id_column="item_id", timestamp_column="timestamp"
        )

    def _update_dataset(
        self, history: TimeSeriesDataFrame, forecast: TimeSeriesDataFrame
    ) -> TimeSeriesDataFrame:
        """
        Append forecasts to training data for the next iteration.

        Args:
            history (TimeSeriesDataFrame): Historical time series data.
            forecast (TimeSeriesDataFrame): Forecasted time series data.

        Returns:
            TimeSeriesDataFrame: Updated time series data.
        """
        new_data = forecast.reset_index()[["item_id", "timestamp", "0.5"]]
        new_data = new_data.rename(columns={"0.5": self.config["target"]})
        updated_df = pd.concat([history.reset_index(), new_data])
        return TimeSeriesDataFrame.from_data_frame(
            updated_df, id_column="item_id", timestamp_column="timestamp"
        )

    def _plot_predictions(self, forecasts: Dict[str, pd.DataFrame]) -> None:
        """
        Visualize forecasts with confidence intervals.

        Args:
            forecasts (Dict[str, pd.DataFrame]): Dictionary of forecasts for each valley.
        """
        plt.figure(figsize=(20, 10))
        colors = {"valley1": "blue", "valley2": "orange"}

        for valley_id, data in forecasts.items():
            # Plot historical data
            hist = self.full_history.loc[valley_id].reset_index()
            plt.plot(
                hist["timestamp"],
                hist[self.config["target"]],
                color=colors[valley_id],
                linestyle="--",
                label=f"{valley_id} Actual",
            )

            # Plot forecasts
            fcst = data.reset_index()
            plt.plot(
                fcst["timestamp"],
                fcst["0.5"],
                color=colors[valley_id],
                label=f"{valley_id} Forecast",
            )

            # Confidence interval
            plt.fill_between(
                fcst["timestamp"],
                fcst["0.1"],
                fcst["0.9"],
                color=colors[valley_id],
                alpha=0.2,
                label=f"{valley_id} 80% CI",
            )

        plt.title(
            f"{self.config['target'].title()} Forecast until {self.config['target_year']}"
        )
        plt.xlabel("Year")
        plt.ylabel(self.config["target"].title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.config["output_plot_path"])
        plt.close()

    def run_pipeline(self) -> TimeSeriesDataFrame:
        """
        Main execution flow for the forecasting pipeline.

        Returns:
            TimeSeriesDataFrame: Full history including forecasts.
        """
        # Initialize data
        ts_data = self._load_data()
        self.full_history = ts_data.copy()

        while True:
            # Determine remaining forecast horizon
            last_date = ts_data.index.get_level_values("timestamp").max()
            remaining_years = self.config["target_year"] - last_date.year
            if remaining_years <= 0:
                break

            prediction_window = min(
                self.config["max_prediction_window"], remaining_years
            )
            prediction_length = int(prediction_window * 365)  # Daily data

            # Initialize/retrain predictor
            self.predictor = TimeSeriesPredictor(
                prediction_length=prediction_length,
                target=self.config["target"],
                eval_metric=self.config["metric"],
                quantile_levels=self.config["quantiles"],
            )

            self.predictor.fit(
                ts_data,
                hyperparameters=hyperparameters,
                presets=self.config["model_preset"],
                time_limit=self.config["training_time"],
                verbosity=self.config["log_level"],
            )

                        # Log useful metadata
            logging.info("\n=== AutoGluon Training Summary ===")
            logging.info(f"Best Model: {self.predictor.model_best}")
            logging.info("Leaderboard:")
            logging.info(self.predictor.leaderboard(silent=True).to_string())
            logging.info(f"Fit Summary: {self.predictor.fit_summary()}")

            # Generate future covariates
            future_covariates = self._generate_future_covariates(
                last_date, prediction_length
            )


            # Make predictions
            forecast = self.predictor.predict(
                ts_data, known_covariates=future_covariates
            )

            # Store and update data
            self.full_history = pd.concat([self.full_history, forecast])
            ts_data = self._update_dataset(ts_data, forecast)

        # Visualize results
        self._plot_predictions(
            {vid: self.full_history.loc[vid] for vid in self.config["valley_ids"]}
        )
        return self.full_history


# Configuration Template
CONFIG = {
    "data_path": "data/maurienne_valley_20_years_daily_data.csv",
    "date_column": "Date",
    "target": "Mean Temperature (°C)",
    "target_year": 2050,
    "valley_ids": ["maurienne"],
    "frequency": "D",
    "quantiles": [0.1, 0.5, 0.9],
    "metric": "MASE",
    "model_preset": "high_quality",
    "training_time": 3600,
    "max_prediction_window": 5,  # years
    "output_plot_path": "forecast_plot.png",
    "log_level": 1,
}

CONFIG.update({"model_preset": "medium_quality", "training_time": 1800})  # 30 minutes

# Example Execution
if __name__ == "__main__":
    forecaster = WeatherForecaster(CONFIG)
    results = forecaster.run_pipeline()
    print(f"Forecasting complete. Results saved to {CONFIG['output_plot_path']}")
