# Comparación de Metaheurísticas: TVAC-PSO vs DE/current-to-best/1/bin

Este proyecto implementa y compara dos algoritmos metaheurísticos de optimización continua evaluados sobre la función de Ackley en 3 dimensiones dentro del dominio [-10, 10]^3.

## Requisitos Previos
- Python 3.9 o superior.
- Git.

## Instalación y Configuración (Multiplataforma)

1. **Clonar el repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd metaheuristics_comparison
   ```

2. **Crear y activar el entorno virtual:**
   - **Unix/MacOS:** `python -m venv venv && source venv/bin/activate`
   - **Windows (Command Prompt):** `python -m venv venv` y luego `venv\Scripts\activate.bat`
   - **Windows (PowerShell):** `python -m venv venv` y luego `.\venv\Scripts\Activate.ps1`

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución del Experimento

El proyecto está diseñado para ejecutarse en dos fases distintas mediante argumentos de línea de comandos, garantizando el rigor metodológico de la comparación:

**Fase 1: Sintonización de Parámetros (Llenado de Tabla 2)**
Explora el espacio de hiperparámetros mediante múltiples ejecuciones independientes para encontrar las mejores configuraciones para TVAC-PSO y DE.

    python main.py --phase tune

*Salida esperada:* Archivos CSV en `results/csv_summaries/` con los mejores valores encontrados.

**Fase 2: Comparación y Análisis Dinámico**
Ejecuta ambos algoritmos utilizando únicamente los parámetros óptimos encontrados en la Fase 1, recolectando la telemetría detallada por iteración para el análisis visual.

    python main.py --phase compare

*Salida esperada:* Archivos JSON en `results/json_history/` y renderizado de las gráficas de convergencia, dispersión de población (cajas y bigotes) y radio poblacional.

## Estructura de Salida
Los resultados se depositan en el directorio `results/`:
- `csv_summaries/`: Tablas de resumen para la selección de parámetros.
- `json_history/`: Registros detallados por iteración de las ejecuciones independientes para generación de gráficas de convergencia y dispersión.
