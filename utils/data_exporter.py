import os
import json
import pandas as pd
from typing import List, Dict, Any

class DataExporter:
    """
    Clase utilitaria para manejar la persistencia de datos experimentales.
    Asegura la creación de directorios y formatea los archivos de salida.
    """
    def __init__(self, base_dir: str = "results"):
        self.csv_dir = os.path.join(base_dir, "csv_summaries")
        self.json_dir = os.path.join(base_dir, "json_history")
        
        # Crear los directorios si no existen (evita FileNotFoundError)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)

    def export_tuning_csv(self, results: List[Dict[str, Any]], filename: str) -> str:
        """
        Convierte una lista de resultados de sintonización a un archivo CSV.
        """
        filepath = os.path.join(self.csv_dir, filename)
        
        # Pandas transforma mágicamente la lista de diccionarios en una tabla
        df = pd.DataFrame(results)
        
        # Guardar sin el índice numérico automático de Pandas
        df.to_csv(filepath, index=False)
        
        return filepath

    def export_telemetry_json(self, data: Dict[str, Any], filename: str) -> str:
        """
        Serializa el historial detallado de telemetría a un archivo JSON.
        """
        filepath = os.path.join(self.json_dir, filename)
        
        # Se usa indent=4 para que el JSON sea legible por humanos (pretty-print)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        return filepath