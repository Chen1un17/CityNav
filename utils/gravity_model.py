"""
Gravity Model for Origin-Destination Matrix Generation

This module implements the classic gravity model used in transportation planning
to generate trip distribution between zones based on population, attractiveness,
and spatial impedance functions.

Mathematical Foundation:
T_ij = K * P_i^α * A_j^β * f(C_ij)^γ

Where:
- T_ij: Trips from zone i to zone j
- P_i: Production factor of zone i (e.g., population)
- A_j: Attraction factor of zone j (e.g., employment, population)
- C_ij: Generalized cost/impedance between zones i and j
- f(C_ij): Impedance function (typically exponential decay)
- K: Scaling constant
- α, β, γ: Model parameters

Implementation based on standard transportation planning practices
and MATSim simulation framework requirements.
"""

import numpy as np
import geopandas as gpd
from scipy.spatial.distance import cdist
from typing import Optional, Union, Callable
import warnings


class GravityGenerator:
    """
    Gravity Model for trip generation between spatial zones.
    
    This implementation follows standard transportation engineering practices
    and is designed to be compatible with MATSim simulation requirements.
    """
    
    def __init__(self, 
                 Lambda: float = 0.2,
                 Alpha: float = 0.5, 
                 Beta: float = 0.5,
                 Gamma: float = 0.5,
                 impedance_function: str = 'exponential',
                 distance_decay_parameter: float = 1.0,
                 minimum_distance: float = 1.0,
                 random_seed: Optional[int] = None):
        """
        Initialize the Gravity Model generator.
        
        Parameters:
        -----------
        Lambda : float, default=0.2
            Impedance decay parameter for the friction function f(C_ij)
            Higher values = stronger distance decay effect
            
        Alpha : float, default=0.5
            Production exponent for origin zones (population influence)
            Typical range: 0.5-1.5
            
        Beta : float, default=0.5
            Attraction exponent for destination zones
            Typical range: 0.5-1.5
            
        Gamma : float, default=0.5
            Overall impedance function exponent
            Typical range: 0.5-2.0
            
        impedance_function : str, default='exponential'
            Type of impedance function. Options:
            - 'exponential': f(C) = exp(-Lambda * C)
            - 'power': f(C) = C^(-Lambda)
            - 'gaussian': f(C) = exp(-Lambda * C^2)
            
        distance_decay_parameter : float, default=1.0
            Additional scaling for distance calculations
            
        minimum_distance : float, default=1.0
            Minimum distance threshold to avoid division by zero
            Units should match coordinate system (typically meters)
            
        random_seed : int, optional
            Random seed for reproducibility of stochastic processes
        """
        self.Lambda = Lambda
        self.Alpha = Alpha  
        self.Beta = Beta
        self.Gamma = Gamma
        self.impedance_function = impedance_function
        self.distance_decay_parameter = distance_decay_parameter
        self.minimum_distance = minimum_distance
        
        self.areas = None
        self.centroids = None
        self.distance_matrix = None
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate model parameters are within reasonable bounds."""
        if self.Lambda <= 0:
            raise ValueError("Lambda must be positive")
        if self.Alpha < 0 or self.Beta < 0:
            warnings.warn("Alpha and Beta should typically be positive")
        if self.impedance_function not in ['exponential', 'power', 'gaussian']:
            raise ValueError(f"Unknown impedance function: {self.impedance_function}")
    
    def load_area(self, areas: gpd.GeoDataFrame):
        """
        Load spatial zones for gravity model calculation.
        
        Parameters:
        -----------
        areas : geopandas.GeoDataFrame
            Spatial zones with geometry column.
            Should be in a projected coordinate system (meters).
        """
        if not isinstance(areas, gpd.GeoDataFrame):
            raise TypeError("areas must be a GeoDataFrame")
            
        if areas.crs is None:
            warnings.warn("No CRS specified for areas. Assuming coordinates are in meters.")
        elif areas.crs.is_geographic:
            warnings.warn("Geographic CRS detected. Consider reprojecting to a metric CRS for accurate distance calculations.")
        
        self.areas = areas.copy()
        
        # Calculate zone centroids
        self.centroids = np.array([(geom.centroid.x, geom.centroid.y) 
                                  for geom in self.areas.geometry])
        
        # Compute distance matrix between all zone pairs
        self._compute_distance_matrix()
        
    def _compute_distance_matrix(self):
        """Compute Euclidean distances between all zone centroids."""
        if self.centroids is None:
            raise ValueError("Must load areas first")
            
        # Calculate pairwise distances
        self.distance_matrix = cdist(self.centroids, self.centroids, metric='euclidean')
        
        # Apply minimum distance threshold
        self.distance_matrix = np.maximum(self.distance_matrix, self.minimum_distance)
        
        # Apply distance decay parameter
        if self.distance_decay_parameter != 1.0:
            self.distance_matrix *= self.distance_decay_parameter
    
    def _impedance_function(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply impedance function to distance matrix.
        
        Parameters:
        -----------
        distances : np.ndarray
            Distance matrix
            
        Returns:
        --------
        np.ndarray
            Impedance values (friction factors)
        """
        if self.impedance_function == 'exponential':
            return np.exp(-self.Lambda * distances)
            
        elif self.impedance_function == 'power':
            return np.power(distances, -self.Lambda)
            
        elif self.impedance_function == 'gaussian':
            return np.exp(-self.Lambda * np.power(distances, 2))
            
        else:
            raise ValueError(f"Unknown impedance function: {self.impedance_function}")
    
    def generate(self, 
                 population: np.ndarray,
                 attractions: Optional[np.ndarray] = None,
                 production_factors: Optional[np.ndarray] = None,
                 balancing_iterations: int = 10,
                 balancing_tolerance: float = 0.01) -> np.ndarray:
        """
        Generate Origin-Destination matrix using gravity model.
        
        Parameters:
        -----------
        population : np.ndarray
            Population or trip production potential for each zone
            Shape: (n_zones,)
            
        attractions : np.ndarray, optional
            Attraction factors for each zone. If None, uses population.
            Shape: (n_zones,)
            
        production_factors : np.ndarray, optional
            Additional production scaling factors. If None, uses population.
            Shape: (n_zones,)
            
        balancing_iterations : int, default=10
            Number of iterations for doubly-constrained balancing
            
        balancing_tolerance : float, default=0.01
            Convergence tolerance for balancing procedure
            
        Returns:
        --------
        np.ndarray
            Origin-Destination matrix
            Shape: (n_zones, n_zones)
        """
        if self.distance_matrix is None:
            raise ValueError("Must load areas first")
            
        population = np.asarray(population, dtype=np.float64)
        n_zones = len(population)
        
        if len(population) != len(self.areas):
            raise ValueError(f"Population array length ({len(population)}) must match number of areas ({len(self.areas)})")
        
        # Set default values
        if attractions is None:
            attractions = population.copy()
        else:
            attractions = np.asarray(attractions, dtype=np.float64)
            
        if production_factors is None:
            production_factors = population.copy()
        else:
            production_factors = np.asarray(production_factors, dtype=np.float64)
        
        # Validate array shapes
        if len(attractions) != n_zones:
            raise ValueError("Attractions array must have same length as population")
        if len(production_factors) != n_zones:
            raise ValueError("Production factors array must have same length as population")
        
        # Calculate impedance matrix
        impedance_matrix = self._impedance_function(self.distance_matrix)
        
        # Apply Gamma exponent to impedance
        if self.Gamma != 1.0:
            impedance_matrix = np.power(impedance_matrix, self.Gamma)
        
        # Initialize OD matrix with basic gravity model
        # T_ij = K * P_i^α * A_j^β * f(C_ij)^γ
        P_i = np.power(production_factors, self.Alpha)
        A_j = np.power(attractions, self.Beta)
        
        # Create production and attraction matrices
        P_matrix = P_i[:, np.newaxis]  # Column vector
        A_matrix = A_j[np.newaxis, :]  # Row vector
        
        # Calculate base OD matrix
        od_matrix = P_matrix * A_matrix * impedance_matrix
        
        # Set diagonal to zero (no intra-zonal trips in this implementation)
        np.fill_diagonal(od_matrix, 0)
        
        # Apply doubly-constrained balancing if needed
        if balancing_iterations > 0:
            od_matrix = self._balance_matrix(od_matrix, population, attractions, 
                                           balancing_iterations, balancing_tolerance)
        
        # Ensure non-negative values
        od_matrix = np.maximum(od_matrix, 0)
        
        return od_matrix
    
    def _balance_matrix(self, 
                       od_matrix: np.ndarray,
                       productions: np.ndarray,
                       attractions: np.ndarray,
                       max_iterations: int,
                       tolerance: float) -> np.ndarray:
        """
        Apply doubly-constrained balancing to ensure row and column sums match constraints.
        
        This implements the standard iterative proportional fitting procedure
        used in transportation planning.
        
        Parameters:
        -----------
        od_matrix : np.ndarray
            Initial OD matrix
        productions : np.ndarray
            Target row sums (trip productions)
        attractions : np.ndarray
            Target column sums (trip attractions)
        max_iterations : int
            Maximum balancing iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        np.ndarray
            Balanced OD matrix
        """
        balanced_matrix = od_matrix.copy()
        
        # Normalize production and attraction totals
        total_productions = np.sum(productions)
        total_attractions = np.sum(attractions)
        
        if total_productions > 0 and total_attractions > 0:
            # Scale attractions to match productions
            attraction_scale = total_productions / total_attractions
            target_attractions = attractions * attraction_scale
        else:
            target_attractions = attractions.copy()
        
        for iteration in range(max_iterations):
            # Balance row sums (productions)
            row_sums = np.sum(balanced_matrix, axis=1)
            row_factors = np.divide(productions, row_sums, 
                                  out=np.ones_like(productions), 
                                  where=row_sums > 0)
            balanced_matrix = balanced_matrix * row_factors[:, np.newaxis]
            
            # Balance column sums (attractions)
            col_sums = np.sum(balanced_matrix, axis=0)
            col_factors = np.divide(target_attractions, col_sums,
                                  out=np.ones_like(target_attractions),
                                  where=col_sums > 0)
            balanced_matrix = balanced_matrix * col_factors[np.newaxis, :]
            
            # Check convergence
            new_row_sums = np.sum(balanced_matrix, axis=1)
            new_col_sums = np.sum(balanced_matrix, axis=0)
            
            row_error = np.max(np.abs(new_row_sums - productions) / 
                              np.maximum(productions, 1))
            col_error = np.max(np.abs(new_col_sums - target_attractions) / 
                              np.maximum(target_attractions, 1))
            
            if row_error < tolerance and col_error < tolerance:
                break
        
        return balanced_matrix
    
    def calculate_trip_statistics(self, od_matrix: np.ndarray) -> dict:
        """
        Calculate summary statistics for the generated OD matrix.
        
        Parameters:
        -----------
        od_matrix : np.ndarray
            Origin-Destination matrix
            
        Returns:
        --------
        dict
            Dictionary containing various statistics
        """
        total_trips = np.sum(od_matrix)
        mean_trip_length = np.sum(od_matrix * self.distance_matrix) / total_trips if total_trips > 0 else 0
        
        row_sums = np.sum(od_matrix, axis=1)  # Productions
        col_sums = np.sum(od_matrix, axis=0)  # Attractions
        
        # Calculate distribution metrics
        non_zero_trips = od_matrix[od_matrix > 0]
        
        stats = {
            'total_trips': total_trips,
            'mean_trip_length': mean_trip_length,
            'min_trip_length': np.min(self.distance_matrix[od_matrix > 0]) if len(non_zero_trips) > 0 else 0,
            'max_trip_length': np.max(self.distance_matrix[od_matrix > 0]) if len(non_zero_trips) > 0 else 0,
            'total_productions': np.sum(row_sums),
            'total_attractions': np.sum(col_sums),
            'max_production': np.max(row_sums),
            'max_attraction': np.max(col_sums),
            'n_active_od_pairs': np.sum(od_matrix > 0),
            'n_total_od_pairs': od_matrix.shape[0] * od_matrix.shape[1],
            'sparsity': 1 - np.sum(od_matrix > 0) / od_matrix.size,
        }
        
        return stats
    
    def export_matrix(self, 
                     od_matrix: np.ndarray, 
                     filename: str,
                     format: str = 'csv',
                     include_zone_ids: bool = True):
        """
        Export OD matrix to file.
        
        Parameters:
        -----------
        od_matrix : np.ndarray
            Origin-Destination matrix to export
        filename : str
            Output filename
        format : str, default='csv'
            Output format: 'csv', 'numpy', or 'excel'
        include_zone_ids : bool, default=True
            Whether to include zone IDs as row/column labels
        """
        if format == 'csv':
            if include_zone_ids and self.areas is not None:
                import pandas as pd
                zone_ids = [f"zone_{i}" for i in range(len(od_matrix))]
                df = pd.DataFrame(od_matrix, index=zone_ids, columns=zone_ids)
                df.to_csv(filename)
            else:
                np.savetxt(filename, od_matrix, delimiter=',', fmt='%.6f')
                
        elif format == 'numpy':
            np.save(filename, od_matrix)
            
        elif format == 'excel':
            if include_zone_ids and self.areas is not None:
                import pandas as pd
                zone_ids = [f"zone_{i}" for i in range(len(od_matrix))]
                df = pd.DataFrame(od_matrix, index=zone_ids, columns=zone_ids)
                df.to_excel(filename)
            else:
                raise ValueError("Excel export requires zone IDs")
        else:
            raise ValueError(f"Unsupported format: {format}")


def validate_gravity_model_inputs(population: np.ndarray,
                                areas: gpd.GeoDataFrame,
                                model_params: dict) -> tuple:
    """
    Validate inputs for gravity model generation.
    
    Parameters:
    -----------
    population : np.ndarray
        Population data for each zone
    areas : gpd.GeoDataFrame
        Spatial zones
    model_params : dict
        Model parameters
        
    Returns:
    --------
    tuple
        (validated_population, validated_areas, warnings_list)
    """
    warnings_list = []
    
    # Validate population data
    population = np.asarray(population, dtype=np.float64)
    if len(population) != len(areas):
        raise ValueError(f"Population array length ({len(population)}) must match number of areas ({len(areas)})")
    
    if np.any(population < 0):
        warnings_list.append("Negative population values detected")
        
    if np.sum(population) == 0:
        warnings_list.append("Total population is zero - no trips will be generated")
    
    # Validate areas
    if areas.crs is None:
        warnings_list.append("No CRS specified for areas")
    elif areas.crs.is_geographic:
        warnings_list.append("Geographic CRS detected - consider reprojecting to metric system")
    
    # Validate model parameters
    required_params = ['Lambda', 'Alpha', 'Beta', 'Gamma']
    for param in required_params:
        if param not in model_params:
            raise ValueError(f"Missing required parameter: {param}")
        if not isinstance(model_params[param], (int, float)):
            raise ValueError(f"Parameter {param} must be numeric")
    
    return population, areas, warnings_list