import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class MetaheuristicBase(ABC):
    """
    Clase abstracta base para algoritmos de optimización metaheurística.
    Define el contrato de ejecución y maneja la recolección de telemetría.
    """
    def __init__(self, objective_func: Any, bounds: Tuple[float, float], pop_size: int, dimensions: int, max_iter: int):
        self.objective_func = objective_func
        self.lower_bound, self.upper_bound = bounds
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.max_iter = max_iter
        
        # Estructura de telemetría para la fase de comparación
        self.history = {
            "best_historical_fitness": [],
            "best_iter_fitness": [],
            "population_radius": [],
            "population_positions": [] # Opcional: llenar solo para la mejor ejecución
        }

    def _initialize_population(self) -> np.ndarray:
        """
        Genera la matriz inicial de población con distribución uniforme continua.
        """
        return np.random.uniform(
            low=self.lower_bound, 
            high=self.upper_bound, 
            size=(self.pop_size, self.dimensions)
        )
        
    def _calculate_population_radius(self, population: np.ndarray) -> float:
        """
        Calcula el radio poblacional zeta definido como la distancia Euclidiana máxima
        entre cualquier individuo y la media poblacional.
        """
        mu = np.mean(population, axis=0)
        distances = np.linalg.norm(population - mu, axis=1)
        return float(np.max(distances))

    @abstractmethod
    def optimize(self) -> Dict[str, Any]:
        """
        Método principal de optimización.
        Debe ser sobrescrito por las implementaciones concretas (PSO, DE).
        
        Returns:
            Dict: Diccionario con la mejor solución, su fitness y el historial.
        """
        pass