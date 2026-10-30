import os
import json
import numpy as np
import pandas as pd
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
        """Genera las gráficas separadas: (a) histórico y (b) iteración."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Gráfica (a): Mejor solución histórica f(x*)
        ax1.plot(pso_data["history"]["best_historical_fitness"], label="PSO", color="#1f77b4", linestyle="-")
        ax1.plot(de_data["history"]["best_historical_fitness"], label="DE", color="#ff7f0e", linestyle="-")
        ax1.set_title("(a) Mejor solución encontrada hasta el momento f(x*)")
        ax1.set_ylabel("Valor de la Función")
        ax1.legend()
        
        # Gráfica (b): Mejor solución en la población por iteración f(x+)
        ax2.plot(pso_data["history"]["best_iter_fitness"], label="PSO", color="#1f77b4", linestyle="--")
        ax2.plot(de_data["history"]["best_iter_fitness"], label="DE", color="#ff7f0e", linestyle="--")
        ax2.set_title("(b) Mejor solución en la población por iteración f(x+)")
        ax2.set_xlabel("Iteración (t)")
        ax2.set_ylabel("Valor de la Función")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "grafica_a_b_convergencia.png"))
        plt.close()

    def plot_population_radius(self, de_data: dict, pso_data: dict):
        """Genera la gráfica (d) del radio poblacional ζ."""
        plt.figure(figsize=(10, 5))
        plt.plot(pso_data["history"]["population_radius"], label="PSO", color="#1f77b4")
        plt.plot(de_data["history"]["population_radius"], label="DE", color="#ff7f0e")
        
        plt.title("(d) Radio Poblacional (ζ) por Iteración")
        plt.xlabel("Iteración (t)")
        plt.ylabel("Norma Euclidiana (ζ)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "grafica_d_radio_poblacional.png"))
        plt.close()

    def plot_boxplots(self, data: dict, algo_name: str):
        """Genera la gráfica (c) de caja y bigotes agrupando 3 dimensiones por iteración."""
        positions = np.array(data["history"]["population_positions"])
        iters, pop_size, dims = positions.shape
        
        # Aplanar los datos para Seaborn
        records = []
        for t in range(iters):
            for d in range(dims):
                for i in range(pop_size):
                    records.append({
                        "Iteración": t,
                        "Dimensión": f"Dim {d+1}",
                        "Valor": positions[t, i, d]
                    })
        
        df = pd.DataFrame(records)
        
        plt.figure(figsize=(16, 6))
        sns.boxplot(
            data=df, 
            x="Iteración", 
            y="Valor", 
            hue="Dimensión", 
            fliersize=2, 
            linewidth=0.8,
            palette="Set2"
        )
        
        plt.title(f"(c) Dispersión de la Población - {algo_name}")
        plt.xlabel("Iteración (t)")
        plt.ylabel("Valor en el dominio [-10, 10]")
        plt.legend(title="Dimensiones")
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