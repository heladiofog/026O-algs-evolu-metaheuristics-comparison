import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    """Clase utilitaria para generar gráficas comparativas de telemetría."""
    
    def __init__(self, base_dir: str = "results"):
        self.json_dir = os.path.join(base_dir, "json_history")
        self.plots_dir = os.path.join(base_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")

    def _load_json(self, algo_name: str) -> dict:
        path = os.path.join(self.json_dir, f"telemetry_{algo_name}.json")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def plot_convergence(self, de_data: dict, pso_data: dict):
        """Genera las gráficas (a) mejor histórico y (b) mejor de iteración."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(de_data["history"]["best_historical_fitness"], label="DE Mejor Histórico (a)", color="blue", linestyle="-")
        plt.plot(de_data["history"]["best_iter_fitness"], label="DE Mejor Población (b)", color="lightblue", linestyle="--")
        
        plt.plot(pso_data["history"]["best_historical_fitness"], label="PSO Mejor Histórico (a)", color="red", linestyle="-")
        plt.plot(pso_data["history"]["best_iter_fitness"], label="PSO Mejor Población (b)", color="salmon", linestyle="--")
        
        plt.title("Convergencia: Función de Ackley 3D")
        plt.xlabel("Iteración")
        plt.ylabel("Valor de la Función Objetivo")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "grafica_a_b_convergencia.png"))
        plt.close()

    def plot_population_radius(self, de_data: dict, pso_data: dict):
        """Genera la gráfica (d) del radio poblacional ζ."""
        plt.figure(figsize=(10, 6))
        plt.plot(de_data["history"]["population_radius"], label="DE", color="blue")
        plt.plot(pso_data["history"]["population_radius"], label="PSO", color="red")
        
        plt.title("Radio Poblacional (ζ) por Iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Norma Euclidiana (ζ)")
        plt.legend()
        plt.savefig(os.path.join(self.plots_dir, "grafica_d_radio_poblacional.png"))
        plt.close()

    def plot_boxplots(self, data: dict, algo_name: str):
        """Genera la gráfica (c) de caja y bigotes para la dispersión poblacional."""
        positions = np.array(data["history"]["population_positions"])
        iters, pop, dims = positions.shape
        
        fig, axes = plt.subplots(dims, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Dispersión de Población por Dimensión - {algo_name}")
        
        for d in range(dims):
            dim_data = [positions[i, :, d] for i in range(iters)]
            sns.boxplot(data=dim_data, ax=axes[d], fliersize=3, palette="Set2")
            axes[d].set_ylabel(f"Dim {d+1}")
            
        axes[-1].set_xlabel("Iteración")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f"grafica_c_cajas_{algo_name}.png"))
        plt.close()

    def generate_all(self):
        de_data = self._load_json("DE")
        pso_data = self._load_json("PSO")
        
        self.plot_convergence(de_data, pso_data)
        self.plot_population_radius(de_data, pso_data)
        self.plot_boxplots(de_data, "DE")
        self.plot_boxplots(pso_data, "PSO")