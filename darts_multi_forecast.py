import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import os
from typing import Dict, List, Optional, Union, Tuple
from darts.models import NBEATSModel, BlockRNNModel, TCNModel, TFTModel
from darts.metrics import mape, mae, mase
from darts import TimeSeries
from darts.utils.likelihood_models import GaussianLikelihood
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

# Configure logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logging.basicConfig(
    filename=os.path.join(log_dir, 'darts_training_alpine_valley.log'),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Set up PyTorch for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    from torch.cuda.amp import GradScaler
    scaler = GradScaler(enabled=True)

class WeatherForecaster:
    def __init__(self, config: Dict) -> None:
        """
        Initialize the WeatherForecaster with the given configuration.

        Args:
            config (Dict): Configuration dictionary containing parameters for data loading,
                           model training, and forecasting.
        """
        self.config = config
        self.model = None
        self.scalers = {}
        self.full_history = None
        self.covariates = None

    def _load_data(self) -> Dict[str, TimeSeries]:
        """
        Load and preprocess time series data.

        Returns:
            Dict[str, TimeSeries]: Dictionary mapping valley IDs to TimeSeries objects.
        """
        df = pd.read_csv(self.config["data_path"])
        df["timestamp"] = pd.to_datetime(df[self.config["date_column"]])

        series_dict = {}
        for valley_id in self.config["valley_ids"]:
            # Filter data for this valley if needed
            valley_df = df[df["item_id"] == valley_id] if "item_id" in df.columns else df

            # Select target columns
            target_cols = self.config["target_columns"]
            valley_df = valley_df[["timestamp"] + target_cols]

            # Convert to Darts TimeSeries
            ts = TimeSeries.from_dataframe(
                valley_df,
                time_col="timestamp",
                value_cols=target_cols
            )
            series_dict[valley_id] = ts

        return series_dict

    def _generate_covariates(self, series_dict: Dict[str, TimeSeries]) -> Dict[str, TimeSeries]:
        """
        Generate covariates for forecasting such as year, month, and day of year.

        Args:
            series_dict (Dict[str, TimeSeries]): Dictionary of time series data.

        Returns:
            Dict[str, TimeSeries]: Dictionary of covariate time series.
        """
        covariates_dict = {}
        for valley_id, ts in series_dict.items():
            # Create year and month covariates
            year_series = datetime_attribute_timeseries(ts, attribute="year")
            month_series = datetime_attribute_timeseries(ts, attribute="month")
            day_of_year_series = datetime_attribute_timeseries(ts, attribute="dayofyear")

            # Normalize covariates
            scaler = Scaler()
            year_series = scaler.fit_transform(year_series)
            month_series = scaler.fit_transform(month_series)
            day_of_year_series = scaler.fit_transform(day_of_year_series)

            # Stack covariates
            covariates = year_series.stack(month_series).stack(day_of_year_series)
            covariates_dict[valley_id] = covariates

        return covariates_dict

    def _preprocess_data(self, series_dict: Dict[str, TimeSeries]) -> Tuple[Dict[str, TimeSeries], Dict[str, TimeSeries]]:
        """
        Preprocess time series data, including scaling and train/validation split.

        Args:
            series_dict (Dict[str, TimeSeries]): Dictionary of time series data.

        Returns:
            Tuple[Dict[str, TimeSeries], Dict[str, TimeSeries]]: Dictionaries of scaled training and validation series.
        """
        train_dict = {}
        val_dict = {}

        for valley_id, ts in series_dict.items():
            # Split into training and validation
            val_length = self.config.get("validation_length", 24)  # 2 years of monthly data by default
            train_ts, val_ts = ts[:-val_length], ts[-val_length:]

            # Scale the data
            scaler = Scaler()
            train_ts_scaled = scaler.fit_transform(train_ts)
            val_ts_scaled = scaler.transform(val_ts)

            # Store the scaler for later use
            self.scalers[valley_id] = scaler

            train_dict[valley_id] = train_ts_scaled
            val_dict[valley_id] = val_ts_scaled

        return train_dict, val_dict

    def _create_model(self) -> Union[NBEATSModel, BlockRNNModel, TCNModel, TFTModel]:
        """
        Create and return a forecasting model based on configuration.

        Returns:
            A Darts forecasting model configured for probabilistic forecasting.
        """
        model_type = self.config.get("model_type", "NBEATSModel")
        input_chunk_length = self.config.get("input_chunk_length", 24)
        output_chunk_length = self.config.get("output_chunk_length", 12)
        n_epochs = self.config.get("n_epochs", 100)

        # Common parameters for all models
        common_params = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "n_epochs": n_epochs,
            "random_state": self.config.get("random_state", 42),
            "likelihood": GaussianLikelihood(),  # For probabilistic forecasts
            "force_reset": True,
            "pl_trainer_kwargs": {
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
                "log_every_n_steps": 10,
            }
        }

        if model_type == "NBEATSModel":
            return NBEATSModel(**common_params)
        elif model_type == "BlockRNNModel":
            return BlockRNNModel(
                model=self.config.get("rnn_type", "LSTM"),
                **common_params
            )
        elif model_type == "TCNModel":
            return TCNModel(**common_params)
        elif model_type == "TFTModel":
            return TFTModel(
                hidden_size=self.config.get("hidden_size", 64),
                lstm_layers=self.config.get("lstm_layers", 1),
                num_attention_heads=self.config.get("num_attention_heads", 4),
                dropout=self.config.get("dropout", 0.1),
                **common_params
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _train_model(self, train_dict: Dict[str, TimeSeries], val_dict: Dict[str, TimeSeries]) -> None:
        """
        Train the model on the given data.

        Args:
            train_dict (Dict[str, TimeSeries]): Dictionary of training time series.
            val_dict (Dict[str, TimeSeries]): Dictionary of validation time series.
        """
        # Create model
        self.model = self._create_model()

        # Convert dictionaries to lists for model.fit()
        train_series = list(train_dict.values())
        val_series = list(val_dict.values())

        # Generate covariates if specified
        if self.config.get("use_covariates", True):
            self.covariates = self._generate_covariates({**train_dict, **val_dict})
            covariates_list = list(self.covariates.values())

            # Train with covariates
            self.model.fit(
                series=train_series,
                past_covariates=covariates_list,
                val_series=val_series,
                val_past_covariates=covariates_list,
                verbose=True
            )
        else:
            # Train without covariates
            self.model.fit(
                series=train_series,
                val_series=val_series,
                verbose=True
            )

        logging.info(f"Model training completed: {self.model}")

    def _forecast(self, train_dict: Dict[str, TimeSeries], horizon: int) -> Dict[str, TimeSeries]:
        """
        Generate forecasts for the specified horizon.

        Args:
            train_dict (Dict[str, TimeSeries]): Dictionary of training time series.
            horizon (int): Forecast horizon in time steps.

        Returns:
            Dict[str, TimeSeries]: Dictionary of forecast time series with probabilistic samples.
        """
        forecasts = {}

        for valley_id, ts in train_dict.items():
            if self.config.get("use_covariates", True) and self.covariates:
                # Forecast with covariates
                forecast = self.model.predict(
                    n=horizon,
                    series=ts,
                    past_covariates=self.covariates[valley_id],
                    num_samples=100  # For probabilistic forecasts
                )
            else:
                # Forecast without covariates
                forecast = self.model.predict(
                    n=horizon,
                    series=ts,
                    num_samples=100  # For probabilistic forecasts
                )

            # Inverse transform the forecast
            forecast = self.scalers[valley_id].inverse_transform(forecast)
            forecasts[valley_id] = forecast

        return forecasts

    def _plot_forecasts(self,
                        full_series: Dict[str, TimeSeries],
                        forecasts: Dict[str, TimeSeries],
                        val_series: Optional[Dict[str, TimeSeries]] = None) -> None:
        """
        Plot forecasts with confidence intervals for each feature.

        Args:
            full_series (Dict[str, TimeSeries]): Dictionary of full historical time series.
            forecasts (Dict[str, TimeSeries]): Dictionary of forecast time series.
            val_series (Optional[Dict[str, TimeSeries]]): Dictionary of validation time series.
        """
        feature_colors = {
            "Mean Temperature (°C)": "red",
            "Min Temperature (°C)": "blue",
            "Max Temperature (°C)": "green",
            # Add more features as needed
        }

        for valley_id in full_series.keys():
            # Get feature names for this valley
            feature_names = full_series[valley_id].components

            # Plot each feature separately
            for i, feature in enumerate(feature_names):
                plt.figure(figsize=(15, 8))

                # Plot historical data
                hist_series = full_series[valley_id].univariate_component(i)
                hist_series.plot(label=f"{valley_id} - {feature} (Actual)", color=feature_colors.get(feature, "blue"))

                # Plot validation data if provided
                if val_series and valley_id in val_series:
                    val_data = val_series[valley_id].univariate_component(i)
                    val_data.plot(label=f"{valley_id} - {feature} (Validation)", color="purple", linestyle="--")

                # Plot forecast with confidence intervals
                forecast = forecasts[valley_id].univariate_component(i)
                forecast.plot(label=f"{valley_id} - {feature} (Forecast)", color="orange")

                # Plot confidence intervals if this is a probabilistic forecast
                if forecast.n_samples > 1:
                    p10, p90 = forecast.quantiles_timeseries(0.1, 0.9)
                    plt.fill_between(
                        forecast.time_index,
                        p10.values().flatten(),
                        p90.values().flatten(),
                        alpha=0.2,
                        color="orange",
                        label="80% Confidence Interval"
                    )

                plt.title(f"{feature} Forecast for {valley_id} Valley until {self.config['target_year']}")
                plt.xlabel("Year")
                plt.ylabel(feature)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{valley_id}_{feature.replace(' ', '_')}_forecast.png")
                plt.close()

                logging.info(f"Plot saved for {valley_id} - {feature}")

    def run_pipeline(self) -> Dict[str, TimeSeries]:
        """
        Main execution flow for the forecasting pipeline.

        Returns:
            Dict[str, TimeSeries]: Dictionary of forecast time series.
        """
        # Load data
        series_dict = self._load_data()
        self.full_history = series_dict

        # Preprocess data
        train_dict, val_dict = self._preprocess_data(series_dict)

        # Train model
        self._train_model(train_dict, val_dict)

        # Calculate forecast horizon
        sample_ts = list(series_dict.values())[0]
        forecast_end_year = self.config["target_year"]
        current_end_year = sample_ts.time_index[-1].year
        years_to_forecast = forecast_end_year - current_end_year

        # Convert years to number of time steps based on data frequency
        steps_per_year = 12  # Assuming monthly data
        if self.config.get("frequency") == "D":
            steps_per_year = 365
        horizon = years_to_forecast * steps_per_year

        # Generate forecasts
        forecasts = self._forecast(train_dict, horizon)

        # Plot results
        self._plot_forecasts(self.full_history, forecasts, val_dict)

        return forecasts


# Example configuration
CONFIG = {
    "data_path": "data/maurienne_valley_20_years_daily_data.csv",
    "date_column": "Date",
    "target_columns": ["Mean Temperature (°C)", "Min Temperature (°C)", "Max Temperature (°C)"],
    "target_year": 2050,
    "valley_ids": ["maurienne"],
    "frequency": "D",  # "D" for daily, "M" for monthly
    "model_type": "NBEATSModel",  # Options: "NBEATSModel", "BlockRNNModel", "TCNModel", "TFTModel"
    "input_chunk_length": 365,  # Lookback window (1 year of daily data)
    "output_chunk_length": 30,  # Output window (1 month of daily data)
    "n_epochs": 100,
    "use_covariates": True,
    "validation_length": 365 * 2,  # 2 years of daily data
    "random_state": 42,
}

# Example execution
if __name__ == "__main__":
    forecaster = WeatherForecaster(CONFIG)
    results = forecaster.run_pipeline()
    print(f"Forecasting complete. Results saved to .png files")
