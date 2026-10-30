import numpy as np
import itertools
from typing import List, Dict, Any, Type
from core.metaheuristic_base import MetaheuristicBase

class ParameterTuner:
    """
    Orquestador para la sintonización de hiperparámetros mediante Grid Search aleatorio.
    Ejecuta múltiples pruebas independientes para garantizar validez estadística.
    """
    def __init__(self, objective_func: Any, bounds: Tuple[float, float], pop_size: int, 
                 dimensions: int, max_iter: int, independent_runs: int = 30):
        self.objective_func = objective_func
        self.bounds = bounds
        self.pop_size = pop_size
        self.dimensions = dimensions
        self.max_iter = max_iter
        self.runs = independent_runs

    def _generate_de_variants(self) -> List[Dict[str, Any]]:
        """
        Genera 9 variantes (3x3) para Evolución Diferencial.
        F en [0.4, 0.9], p_c en [0.1, 0.9]
        """
        f_vals = np.random.uniform(0.4, 0.9, 3)
        pc_vals = np.random.uniform(0.1, 0.9, 3)
        
        variants = []
        for f, pc in itertools.product(f_vals, pc_vals):
            variants.append({
                "f_weight": round(float(f), 4),
                "p_crossover": round(float(pc), 4)
            })
        return variants

    def _generate_pso_variants(self) -> List[Dict[str, Any]]:
        """
        Genera 12 variantes (2x2x3) para TVAC-PSO.
        c1 decae (exploración a explotación), c2 crece (explotación final), w fijo por variante.
        """
        # Generar 2 pares para c1: inicial alto, final bajo
        c1_pairs = [
            (round(float(np.random.uniform(2.0, 2.5)), 4), round(float(np.random.uniform(0.5, 1.0)), 4))
            for _ in range(2)
        ]
        # Generar 2 pares para c2: inicial bajo, final alto
        c2_pairs = [
            (round(float(np.random.uniform(0.5, 1.0)), 4), round(float(np.random.uniform(2.0, 2.5)), 4))
            for _ in range(2)
        ]
        # Generar 3 valores para w
        w_vals = np.random.uniform(0.4, 0.9, 3)
        
        variants = []
        for c1, c2, w in itertools.product(c1_pairs, c2_pairs, w_vals):
            variants.append({
                "c1_bounds": c1,
                "c2_bounds": c2,
                "w": round(float(w), 4)
            })
        return variants

    def tune_algorithm(self, algorithm_class: Type[MetaheuristicBase], algorithm_name: str) -> List[Dict[str, Any]]:
        """
        Evalúa todas las variantes generadas para una metaheurística ejecutando múltiples corridas independientes.
        """
        if algorithm_name == "DE":
            variants = self._generate_de_variants()
        elif algorithm_name == "PSO":
            variants = self._generate_pso_variants()
        else:
            raise ValueError("Algoritmo no soportado para sintonización.")

        results = []
        
        for index, kwargs in enumerate(variants):
            variant_id = f"{algorithm_name}_var_{index+1}"
            final_fitnesses = []
            
            # Ejecuciones independientes por cada configuración de parámetros
            for run in range(self.runs):
                algo = algorithm_class(
                    objective_func=self.objective_func,
                    bounds=self.bounds,
                    pop_size=self.pop_size,
                    dimensions=self.dimensions,
                    max_iter=self.max_iter,
                    **kwargs
                )
                
                execution_result = algo.optimize()
                final_fitnesses.append(execution_result["best_fitness"])
            
            # Cálculo estadístico para la configuración actual
            mean_fitness = np.mean(final_fitnesses)
            std_fitness = np.std(final_fitnesses)
            
            # Consolidar registro
            record = {
                "Algoritmo": algorithm_name,
                "ID_Variante": variant_id,
                **kwargs,
                "Media_Fitness_Final": round(float(mean_fitness), 6),
                "Desv_Estandar": round(float(std_fitness), 6)
            }
            
            # Desacoplar las tuplas de PSO para aplanar el CSV final
            if "c1_bounds" in record:
                record["c1_i"], record["c1_f"] = record.pop("c1_bounds")
                record["c2_i"], record["c2_f"] = record.pop("c2_bounds")
                
            results.append(record)
            
        return results