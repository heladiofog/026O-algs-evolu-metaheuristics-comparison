import numpy as np

class AckleyFunction:
    """
    Implementación vectorizada de la función de Ackley para n dimensiones.
    Minimo global en x = [0, 0, ..., 0] donde f(x) = 0.
    """
    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi):
        self.a = a
        self.b = b
        self.c = c

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evalúa un conjunto de soluciones en la función de Ackley.
        
        Args:
            x (np.ndarray): Matriz bidimensional de forma (poblacion, dimensiones).
                            Para una sola partícula, pasar forma (1, dimensiones).
        
        Returns:
            np.ndarray: Vector unidimensional con los valores de aptitud (fitness).
        """
        d = x.shape[1]
        
        sum_sq = np.sum(x**2, axis=1)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        
        sum_cos = np.sum(np.cos(self.c * x), axis=1)
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + self.a + np.exp(1)