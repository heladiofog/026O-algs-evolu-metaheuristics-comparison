import numpy as np
from core.metaheuristic_base import MetaheuristicBase
from typing import Dict, Any, Tuple

class DECurrentToBest(MetaheuristicBase):
    """
    Evolución Diferencial con estrategia DE/current-to-best/1/bin.
    """
    def __init__(self, objective_func: Any, bounds: Tuple[float, float], pop_size: int, 
                 dimensions: int, max_iter: int, f_weight: float, p_crossover: float):
        super().__init__(objective_func, bounds, pop_size, dimensions, max_iter)
        self.F = f_weight
        self.p_c = p_crossover

    def optimize(self) -> Dict[str, Any]:
        # 1. Inicialización
        positions = self._initialize_population()
        fitness = self.objective_func.evaluate(positions)
        
        # 2. Identificar al mejor individuo inicial (x_best)
        best_idx = np.argmin(fitness)
        best_position = np.copy(positions[best_idx])
        best_fitness = fitness[best_idx]
        
        # Limpiar el historial para esta ejecución
        self.history = {k: [] for k in self.history}

        # 3. Ciclo evolutivo
        for t in range(self.max_iter):
            # Preparar arreglos para los índices mutuamente excluyentes (r1, r2)
            r1_idx = np.zeros(self.pop_size, dtype=int)
            r2_idx = np.zeros(self.pop_size, dtype=int)
            
            for i in range(self.pop_size):
                # Generar conjunto de candidatos excluyendo al índice actual 'i'
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1_idx[i], r2_idx[i] = np.random.choice(candidates, 2, replace=False)
            
            # 4. Mutación: current-to-best/1 vectorizada
            # v_i = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            v = (positions + 
                 self.F * (best_position - positions) + 
                 self.F * (positions[r1_idx] - positions[r2_idx]))
            
            # 5. Cruce: Binomial (bin)
            # Crear máscara booleana donde rand <= p_c
            cross_mask = np.random.rand(self.pop_size, self.dimensions) <= self.p_c
            
            # Garantizar que al menos una dimensión mute (j_rand)
            j_rand = np.random.randint(0, self.dimensions, size=self.pop_size)
            cross_mask[np.arange(self.pop_size), j_rand] = True
            
            # Construir el vector de prueba 'u' usando la máscara y aplicar límites
            u = np.where(cross_mask, v, positions)
            u = np.clip(u, self.lower_bound, self.upper_bound)
            
            # 6. Selección (Supervivencia del más apto)
            u_fitness = self.objective_func.evaluate(u)
            
            # Si el vector de prueba es menor o igual, reemplaza al actual
            better_mask = u_fitness <= fitness
            positions[better_mask] = u[better_mask]
            fitness[better_mask] = u_fitness[better_mask]
            
            # Actualizar el mejor global (x_best)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_position = np.copy(positions[current_best_idx])
                best_fitness = fitness[current_best_idx]
            
            # 7. Registro de telemetría por iteración
            self.history["best_historical_fitness"].append(float(best_fitness))
            self.history["best_iter_fitness"].append(float(np.min(fitness)))
            self.history["population_radius"].append(self._calculate_population_radius(positions))
            
            # (Opcional) Guardar matriz 30x3 completa para el gráfico de caja y bigotes de la iteración final
            self.history["population_positions"].append(positions.tolist())

        return {
            "best_position": best_position.tolist(),
            "best_fitness": float(best_fitness),
            "history": self.history
        }