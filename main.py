import argparse
import pandas as pd
import os
from core.ackley_function import AckleyFunction
from core.parameter_tuner import ParameterTuner
from algorithms.tvac_pso import TVAC_PSO
from algorithms.de_current_to_best import DECurrentToBest
from utils.data_exporter import DataExporter

# Configuraciones globales del experimento dictadas por los requerimientos
BOUNDS = (-10.0, 10.0)
POP_SIZE = 30
DIMENSIONS = 3
MAX_ITER = 30

def run_tuning():
    """Fase 1: Ejecuta Grid Search para DE y PSO, y exporta resúmenes CSV."""
    print("Iniciando Fase 1: Sintonización de Parámetros...")
    func = AckleyFunction()
    tuner = ParameterTuner(func, BOUNDS, POP_SIZE, DIMENSIONS, MAX_ITER, independent_runs=30)
    exporter = DataExporter()

    # Sintonizar Evolución Diferencial
    print("Sintonizando DE/current-to-best/1/bin...")
    de_results = tuner.tune_algorithm(DECurrentToBest, "DE")
    de_csv = exporter.export_tuning_csv(de_results, "tune_summary_DE.csv")
    print(f"Resultados DE guardados en: {de_csv}")

    # Sintonizar TVAC-PSO
    print("Sintonizando TVAC-PSO...")
    pso_results = tuner.tune_algorithm(TVAC_PSO, "PSO")
    pso_csv = exporter.export_tuning_csv(pso_results, "tune_summary_PSO.csv")
    print(f"Resultados PSO guardados en: {pso_csv}")
    print("Fase 1 completada con éxito. Consulta los CSV para llenar la Tabla 2.")

def run_comparison():
    """Fase 2: Lee los mejores parámetros, ejecuta 1 vez y guarda telemetría JSON."""
    print("Iniciando Fase 2: Comparación de Heurísticas...")
    exporter = DataExporter()
    func = AckleyFunction()
    
    # Rutas esperadas de los CSV
    de_csv_path = os.path.join(exporter.csv_dir, "tune_summary_DE.csv")
    pso_csv_path = os.path.join(exporter.csv_dir, "tune_summary_PSO.csv")
    
    if not os.path.exists(de_csv_path) or not os.path.exists(pso_csv_path):
        print("Error: No se encontraron los archivos CSV. Ejecuta primero '--phase tune'.")
        return

    # 1. Extraer los mejores parámetros (los que tengan menor Media_Fitness_Final)
    de_df = pd.read_csv(de_csv_path)
    best_de_row = de_df.loc[de_df["Media_Fitness_Final"].idxmin()]
    
    pso_df = pd.read_csv(pso_csv_path)
    best_pso_row = pso_df.loc[pso_df["Media_Fitness_Final"].idxmin()]

    print(f"Mejor config DE encontrada: F={best_de_row['f_weight']}, p_c={best_de_row['p_crossover']}")
    
    # 2. Instanciar y ejecutar DE
    de_algo = DECurrentToBest(
        objective_func=func, bounds=BOUNDS, pop_size=POP_SIZE, dimensions=DIMENSIONS, 
        max_iter=MAX_ITER, 
        f_weight=best_de_row["f_weight"], p_crossover=best_de_row["p_crossover"]
    )
    de_telemetry = de_algo.optimize()
    de_json = exporter.export_telemetry_json(de_telemetry, "telemetry_DE.json")
    print(f"Telemetría de DE guardada en: {de_json}")

    # 3. Instanciar y ejecutar PSO
    pso_algo = TVAC_PSO(
        objective_func=func, bounds=BOUNDS, pop_size=POP_SIZE, dimensions=DIMENSIONS, 
        max_iter=MAX_ITER,
        c1_bounds=(best_pso_row["c1_i"], best_pso_row["c1_f"]),
        c2_bounds=(best_pso_row["c2_i"], best_pso_row["c2_f"]),
        w=best_pso_row["w"]
    )
    pso_telemetry = pso_algo.optimize()
    pso_json = exporter.export_telemetry_json(pso_telemetry, "telemetry_PSO.json")
    print(f"Telemetría de PSO guardada en: {pso_json}")
    print("Fase 2 completada. JSONs listos para renderizar gráficas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparador de Metaheurísticas 3D")
    parser.add_argument("--phase", choices=["tune", "compare"], required=True, 
                        help="Selecciona 'tune' para generar Tabla 2 o 'compare' para extraer gráficas.")
    
    args = parser.parse_args()
    
    if args.phase == "tune":
        run_tuning()
    elif args.phase == "compare":
        run_comparison()