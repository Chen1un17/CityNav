"""
Prediction Engine for Multi-Agent Traffic Control System

Implements autoregressive traffic flow prediction using multiple time windows
to support regional route planning and traffic management decisions.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class EdgeObservation:
    """Single edge observation data."""
    timestamp: float
    vehicle_count: int
    avg_speed: float
    occupancy_rate: float
    congestion_level: int
    eta: float


@dataclass
class PredictionResult:
    """Prediction result for an edge."""
    edge_id: str
    time_window: int
    predicted_vehicle_count: float
    predicted_avg_speed: float
    predicted_congestion_level: int
    confidence: float
    prediction_horizon: float


class AutoregressivePredictor:
    """
    Simple autoregressive predictor for traffic flow metrics.
    
    Uses historical observations to predict future traffic states
    with configurable time windows and horizons.
    """
    
    def __init__(self, window_size: int = 10, alpha: float = 0.3):
        """
        Initialize the predictor.
        
        Args:
            window_size: Number of historical observations to use
            alpha: Exponential smoothing factor (0-1)
        """
        self.window_size = window_size
        self.alpha = alpha
        self.observations: deque = deque(maxlen=window_size)
        self.trend = 0.0
        self.last_prediction = 0.0
        
    def add_observation(self, value: float, timestamp: float):
        """Add a new observation."""
        self.observations.append((value, timestamp))
        
        # Update trend estimation
        if len(self.observations) >= 2:
            recent_values = [obs[0] for obs in list(self.observations)[-3:]]
            if len(recent_values) >= 2:
                self.trend = np.mean(np.diff(recent_values))
    
    def predict(self, horizon_steps: int = 1) -> Tuple[float, float]:
        """
        Predict future value.
        
        Args:
            horizon_steps: Number of steps ahead to predict
            
        Returns:
            Tuple of (predicted_value, confidence)
        """
        if len(self.observations) < 2:
            return 0.0, 0.0
        
        values = [obs[0] for obs in self.observations]
        
        # Exponential smoothing with trend
        if len(values) == 1:
            smoothed = values[0]
        else:
            smoothed = values[0]
            for value in values[1:]:
                smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        
        # Add trend component
        prediction = smoothed + self.trend * horizon_steps
        
        # Calculate confidence based on variance of recent observations
        if len(values) >= 3:
            variance = np.var(values[-5:])  # Use last 5 observations
            confidence = max(0.1, 1.0 / (1.0 + variance))
        else:
            confidence = 0.5
        
        self.last_prediction = prediction
        return max(0.0, prediction), min(1.0, confidence)


class PredictionEngine:
    """
    Traffic flow prediction engine using autoregressive models.
    
    Provides predictions for multiple time windows (180s, 360s, 540s, 720s)
    to support different planning horizons for regional and traffic agents.
    """
    
    def __init__(self, edge_list: List[str], logger, 
                 time_windows: List[int] = None, prediction_horizons: List[int] = None):
        """
        Initialize the prediction engine.
        
        Args:
            edge_list: List of edge IDs to track
            logger: Agent logger instance
            time_windows: Time windows for aggregation (seconds)
            prediction_horizons: Prediction horizons (steps)
        """
        self.edge_list = edge_list
        self.logger = logger
        
        # Default time windows (seconds)
        self.time_windows = time_windows or [180, 360, 540, 720]
        
        # Default prediction horizons (simulation steps ahead)
        self.prediction_horizons = prediction_horizons or [1, 2, 3, 4]
        
        # Data storage for each edge and time window
        self.edge_observations: Dict[str, List[EdgeObservation]] = defaultdict(list)
        self.aggregated_data: Dict[Tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=50)  # Keep last 50 aggregated windows
        )
        
        # Predictors for each edge and metric combination
        self.predictors: Dict[Tuple[str, int, str], AutoregressivePredictor] = {}
        
        # Performance tracking
        self.prediction_accuracy: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)  # Keep last 100 accuracy scores
        )
        
        # Initialize predictors
        self._initialize_predictors()
        
        # Training and update intervals
        self.last_training_time = 0.0
        self.training_interval = 300.0  # Train every 5 minutes
        self.aggregation_intervals = {window: 0.0 for window in self.time_windows}
        
        self.logger.log_info("Prediction Engine initialized with "
                           f"{len(self.edge_list)} edges, "
                           f"{len(self.time_windows)} time windows")
    
    def _initialize_predictors(self):
        """Initialize predictors for all edge-metric combinations."""
        metrics = ['vehicle_count', 'avg_speed', 'congestion_level']
        
        for edge_id in self.edge_list:
            for time_window in self.time_windows:
                for metric in metrics:
                    key = (edge_id, time_window, metric)
                    self.predictors[key] = AutoregressivePredictor()
    
    def update_observations(self, road_info: Dict[str, Dict], current_time: float):
        """
        Update observations with current road information.
        
        Args:
            road_info: Road information dictionary from simulation
            current_time: Current simulation time
        """
        for edge_id in self.edge_list:
            if edge_id in road_info:
                edge_data = road_info[edge_id]
                
                observation = EdgeObservation(
                    timestamp=current_time,
                    vehicle_count=edge_data.get('vehicle_num', 0),
                    avg_speed=edge_data.get('vehicle_speed', 0.0),
                    occupancy_rate=edge_data.get('vehicle_num', 0) / max(1, edge_data.get('road_len', 1) / 8.0),
                    congestion_level=edge_data.get('congestion_level', 0),
                    eta=edge_data.get('eta', 0.0)
                )
                
                self.edge_observations[edge_id].append(observation)
                
                # Keep only recent observations (last hour)
                cutoff_time = current_time - 3600.0
                self.edge_observations[edge_id] = [
                    obs for obs in self.edge_observations[edge_id]
                    if obs.timestamp >= cutoff_time
                ]
        
        # Trigger aggregation for appropriate time windows
        self._aggregate_observations(current_time)
    
    def _aggregate_observations(self, current_time: float):
        """Aggregate observations for different time windows."""
        for time_window in self.time_windows:
            # Check if it's time to aggregate for this window
            if current_time - self.aggregation_intervals[time_window] >= time_window:
                self._aggregate_window(time_window, current_time)
                self.aggregation_intervals[time_window] = current_time
    
    def _aggregate_window(self, time_window: int, current_time: float):
        """Aggregate observations for a specific time window."""
        window_start = current_time - time_window
        
        for edge_id in self.edge_list:
            observations = self.edge_observations[edge_id]
            
            # Filter observations within the time window
            window_obs = [
                obs for obs in observations
                if window_start <= obs.timestamp <= current_time
            ]
            
            if not window_obs:
                continue
            
            # Aggregate metrics
            avg_vehicle_count = np.mean([obs.vehicle_count for obs in window_obs])
            avg_speed = np.mean([obs.avg_speed for obs in window_obs])
            avg_congestion = np.mean([obs.congestion_level for obs in window_obs])
            
            # Store aggregated data
            key = (edge_id, time_window)
            aggregated_point = {
                'timestamp': current_time,
                'vehicle_count': avg_vehicle_count,
                'avg_speed': avg_speed,
                'congestion_level': avg_congestion
            }
            
            self.aggregated_data[key].append(aggregated_point)
            
            # Update predictors with new aggregated data
            predictor_keys = [
                (edge_id, time_window, 'vehicle_count'),
                (edge_id, time_window, 'avg_speed'),
                (edge_id, time_window, 'congestion_level')
            ]
            
            for pred_key in predictor_keys:
                if pred_key in self.predictors:
                    metric_name = pred_key[2]
                    self.predictors[pred_key].add_observation(
                        aggregated_point[metric_name], current_time
                    )
    
    def get_predictions(self, edge_id: str, time_window: int, 
                       horizon_steps: int = 1) -> List[PredictionResult]:
        """
        Get predictions for an edge at a specific time window.
        
        Args:
            edge_id: Edge to predict for
            time_window: Time window for aggregation
            horizon_steps: Number of steps ahead to predict
            
        Returns:
            List of prediction results
        """
        if edge_id not in self.edge_list:
            return []
        
        predictions = []
        metrics = ['vehicle_count', 'avg_speed', 'congestion_level']
        
        for metric in metrics:
            predictor_key = (edge_id, time_window, metric)
            
            if predictor_key in self.predictors:
                predictor = self.predictors[predictor_key]
                predicted_value, confidence = predictor.predict(horizon_steps)
                
                # Convert predicted congestion level to integer
                if metric == 'congestion_level':
                    predicted_congestion = int(round(predicted_value))
                else:
                    predicted_congestion = 0
                
                prediction = PredictionResult(
                    edge_id=edge_id,
                    time_window=time_window,
                    predicted_vehicle_count=predicted_value if metric == 'vehicle_count' else 0,
                    predicted_avg_speed=predicted_value if metric == 'avg_speed' else 0,
                    predicted_congestion_level=predicted_congestion if metric == 'congestion_level' else 0,
                    confidence=confidence,
                    prediction_horizon=horizon_steps * time_window
                )
                
                predictions.append(prediction)
        
        return predictions
    
    def get_batch_predictions(self, edge_ids: List[str], time_window: int = 180,
                            horizon_steps: int = 1) -> Dict[str, List[PredictionResult]]:
        """
        Get predictions for multiple edges at once.
        
        Args:
            edge_ids: List of edge IDs to predict for
            time_window: Time window for aggregation
            horizon_steps: Number of steps ahead to predict
            
        Returns:
            Dictionary mapping edge IDs to prediction results
        """
        results = {}
        
        for edge_id in edge_ids:
            results[edge_id] = self.get_predictions(edge_id, time_window, horizon_steps)
        
        return results
    
    def get_congestion_forecast(self, edge_ids: List[str], 
                              forecast_duration: int = 1800) -> Dict[str, List[float]]:
        """
        Get congestion forecast for multiple edges over a time period.
        
        Args:
            edge_ids: List of edge IDs
            forecast_duration: Duration to forecast (seconds)
            
        Returns:
            Dictionary mapping edge IDs to congestion level forecasts
        """
        forecasts = {}
        time_window = 540  # Use 3-minute windows
        horizon_steps = max(1, forecast_duration // time_window)
        
        for edge_id in edge_ids:
            predictor_key = (edge_id, time_window, 'congestion_level')
            
            if predictor_key in self.predictors:
                predictor = self.predictors[predictor_key]
                
                forecast = []
                for step in range(1, horizon_steps + 1):
                    predicted_value, _ = predictor.predict(step)
                    forecast.append(max(0, min(5, int(round(predicted_value)))))
                
                forecasts[edge_id] = forecast
            else:
                forecasts[edge_id] = [0] * horizon_steps
        
        return forecasts
    
    def train_models(self, current_time: float):
        """
        Train/update prediction models periodically.
        
        Args:
            current_time: Current simulation time
        """
        if current_time - self.last_training_time < self.training_interval:
            return
        
        # For autoregressive models, training is implicit in the update process
        # Here we can perform validation and accuracy assessment
        
        trained_models = 0
        for edge_id in self.edge_list:
            for time_window in self.time_windows:
                key = (edge_id, time_window)
                
                if key in self.aggregated_data and len(self.aggregated_data[key]) >= 5:
                    # Validate recent predictions if available
                    self._validate_predictions(edge_id, time_window)
                    trained_models += 1
        
        self.last_training_time = current_time
        
        if trained_models > 0:
            self.logger.log_info(f"Prediction Engine: Updated {trained_models} model combinations")
    
    def _validate_predictions(self, edge_id: str, time_window: int):
        """Validate recent predictions against actual observations."""
        key = (edge_id, time_window)
        
        if key not in self.aggregated_data or len(self.aggregated_data[key]) < 3:
            return
        
        recent_data = list(self.aggregated_data[key])[-3:]
        
        # Calculate prediction accuracy for the last few points
        for i in range(len(recent_data) - 1):
            actual_data = recent_data[i + 1]
            
            for metric in ['vehicle_count', 'avg_speed', 'congestion_level']:
                predictor_key = (edge_id, time_window, metric)
                
                if predictor_key in self.predictors:
                    predicted_value, _ = self.predictors[predictor_key].predict(1)
                    actual_value = actual_data[metric]
                    
                    if actual_value > 0:  # Avoid division by zero
                        accuracy = 1.0 - abs(predicted_value - actual_value) / max(actual_value, 1.0)
                        accuracy = max(0.0, min(1.0, accuracy))
                        
                        accuracy_key = f"{edge_id}_{time_window}_{metric}"
                        self.prediction_accuracy[accuracy_key].append(accuracy)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get prediction engine performance metrics."""
        if not self.prediction_accuracy:
            return {
                'avg_accuracy': 0.0,
                'total_predictions': 0,
                'active_predictors': len(self.predictors),
                'tracked_edges': len(self.edge_list),
                'time_windows': len(self.time_windows)
            }
        
        # Calculate overall accuracy
        all_accuracies = []
        for accuracy_list in self.prediction_accuracy.values():
            all_accuracies.extend(accuracy_list)
        
        avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        # Count active predictors with observations
        active_predictors = sum(
            1 for predictor in self.predictors.values()
            if len(predictor.observations) > 0
        )
        
        return {
            'avg_accuracy': avg_accuracy,
            'total_predictions': len(all_accuracies),
            'active_predictors': active_predictors,
            'tracked_edges': len(self.edge_list),
            'time_windows': len(self.time_windows),
            'prediction_coverage': active_predictors / len(self.predictors) if self.predictors else 0.0
        }