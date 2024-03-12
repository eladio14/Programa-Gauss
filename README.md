# Sistema de Ecuaciones Lineales con Gauss-Jordan

Este programa resuelve un sistema de ecuaciones lineales utilizando el método de Gauss-Jordan.

## Instalación

No es necesaria la instalación de ninguna biblioteca externa. Sin embargo, si deseas ejecutar el programa en tu computadora, necesitarás instalar Python y las bibliotecas NumPy y Matplotlib.

## Uso

Para utilizar el programa, sigue estos pasos:

1. Define las matrices A y b, donde A es una matriz de coeficientes y b es un vector de términos independientes.

2. Ejecuta la función `gauss_elimination(A, b)` con las matrices A y b como argumentos.

3. La función devuelve la solución del sistema de ecuaciones lineales.

4. Si deseas visualizar la solución en un gráfico 3D, utiliza la función `graficar_solucion(A, b)` con las matrices A y b como argumentos.

### Ejemplo de uso

```python
import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, b):
    # Implementación del algoritmo de Gauss-Jordan
    # ...
    return solucion

def graficar_solucion(A, b):
    # Implementación de la visualización en un gráfico 3D
    # ...
    plt.show()
```
# Ejemplo de uso 1
```python
A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b1 = np.array([8, -11, -3])
solucion1 = gauss_elimination(A1, b1)
graficar_solucion(A1, b1)
```
# Ejemplo de uso 2
```python
A2 = np.array([[1, -1, 1], [2, 1, -1], [3, -2, 2]])
b2 = np.array([2, 1, 5])
solucion2 = gauss_elimination(A2, b2)
graficar_solucion(A2, b2)
```
# Ejemplo de uso 3
```python
A3 = np.array([[3, 1, -2], [2, -2, 4], [1, 1, 3]])
b3 = np.array([7, 6, 13])
solucion3 = gauss_elimination(A3, b3)
graficar_solucion(A3, b3)
```