import numpy as np
from core.metaheuristic_base import MetaheuristicBase
from typing import Dict, Any, Tuple

class TVAC_PSO(MetaheuristicBase):
    """
    Enjambre de Partículas con Coeficientes de Aceleración Variables en el Tiempo (TVAC).
    """
    def __init__(self, objective_func: Any, bounds: Tuple[float, float], pop_size: int, 
                 dimensions: int, max_iter: int, 
                 c1_bounds: Tuple[float, float], c2_bounds: Tuple[float, float], w: float):
        super().__init__(objective_func, bounds, pop_size, dimensions, max_iter)
        self.c1_initial, self.c1_final = c1_bounds
        self.c2_initial, self.c2_final = c2_bounds
        self.w = w

    def optimize(self) -> Dict[str, Any]:
        # 1. Inicialización
        positions = self._initialize_population()
        # Las velocidades suelen inicializarse en 0 o aleatorias; inicializamos en 0 matricialmente
        velocities = np.zeros((self.pop_size, self.dimensions))
        
        # 2. Primera evaluación
        fitness = self.objective_func.evaluate(positions)
        
        # 3. Memoria inicial (pbest y gbest)
        pbest_positions = np.copy(positions)
        pbest_fitness = np.copy(fitness)
        
        gbest_idx = np.argmin(pbest_fitness)
        gbest_position = np.copy(pbest_positions[gbest_idx])
        gbest_fitness = pbest_fitness[gbest_idx]
        
        # Limpiar el historial base para esta ejecución
        self.history = {k: [] for k in self.history}

        # 4. Ciclo evolutivo
        for t in range(self.max_iter):
            # Calcular coeficientes dinámicos para la iteración actual t
            c1_t = self.c1_initial + (self.c1_final - self.c1_initial) * (t / (self.max_iter - 1))
            c2_t = self.c2_initial + (self.c2_final - self.c2_initial) * (t / (self.max_iter - 1))
            
            # Generar variables aleatorias r1, r2 para toda la matriz (pop_size, dimensions)
            r1 = np.random.uniform(0, 1, (self.pop_size, self.dimensions))
            r2 = np.random.uniform(0, 1, (self.pop_size, self.dimensions))
            
            # Actualización vectorizada de velocidades
            velocities = (self.w * velocities + 
                          c1_t * r1 * (pbest_positions - positions) + 
                          c2_t * r2 * (gbest_position - positions))
            
            # Actualización de posiciones y restricción a los límites (clipping)
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
            
            # Evaluar nuevas posiciones
            current_fitness = self.objective_func.evaluate(positions)
            
            # Actualizar pbest (donde el fitness actual sea mejor que el histórico personal)
            improved_indices = current_fitness < pbest_fitness
            pbest_positions[improved_indices] = positions[improved_indices]
            pbest_fitness[improved_indices] = current_fitness[improved_indices]
            
            # Actualizar gbest
            current_gbest_idx = np.argmin(pbest_fitness)
            if pbest_fitness[current_gbest_idx] < gbest_fitness:
                gbest_position = np.copy(pbest_positions[current_gbest_idx])
                gbest_fitness = pbest_fitness[current_gbest_idx]
            
            # Registro de telemetría por iteración
            self.history["best_historical_fitness"].append(float(gbest_fitness))
            self.history["best_iter_fitness"].append(float(np.min(current_fitness)))
            self.history["population_radius"].append(self._calculate_population_radius(positions))
            
            # (Opcional) Almacenar las 30x3 posiciones si se requiere para la gráfica de caja y bigotes
            # self.history["population_positions"].append(positions.tolist())

        return {
            "best_position": gbest_position.tolist(),
            "best_fitness": float(gbest_fitness),
            "history": self.history
        }