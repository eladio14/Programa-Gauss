import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def gauss_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Método de eliminación Gaussiana para resolver un sistema de ecuaciones lineales.

    Parámetros:
        A (np.ndarray): Matriz de coeficientes del sistema.
        b (np.ndarray): Vector de términos independientes.

    Devuelve:
        np.ndarray: Vector de soluciones.
    """
    A = A.astype(float)  # Convertir la matriz A a float64
    b = b.astype(float)  # Convertir el vector b a float64

    n = len(b)

    # Eliminación hacia adelante
    for i in range(n - 1):
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, :] -= factor * A[i, :]
            b[j] -= factor * b[i]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1 :], x[i + 1 :])) / A[i, i]

    return x


def graficar_solucion(A: np.ndarray, b: np.ndarray):
    """
    Función para resolver un sistema de ecuaciones lineales y graficar la solución en 3D.

    Parámetros:
        A (np.ndarray): Matriz de coeficientes del sistema.
        b (np.ndarray): Vector de términos independientes.
    """
    solucion = gauss_elimination(A, b)
    print("Solución del sistema:", solucion)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Puntos de la solución
    x_sol = solucion[0]
    y_sol = solucion[1]
    z_sol = solucion[2]

    # Graficar la solución como un punto
    ax.scatter(
        x_sol,
        y_sol,
        z_sol,
        color="r",
        marker="o",
        s=100,
        label=f"x={x_sol:.2f}, y={y_sol:.2f}, z={z_sol:.2f}",
    )  # Agregamos la leyenda

    # Graficar las ecuaciones
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Ecuaciones del sistema
    Z1 = (b[0] - A[0, 0] * X - A[0, 1] * Y) / A[0, 2]
    Z2 = (b[1] - A[1, 0] * X - A[1, 1] * Y) / A[1, 2]
    Z3 = (b[2] - A[2, 0] * X - A[2, 1] * Y) / A[2, 2]

    ax.plot_surface(
        X, Y, Z1, alpha=0.5, label=f"{A[0,0]}x + {A[0,1]}y + {A[0,2]}z = {b[0]}"
    )  # Agregamos la ecuación
    ax.plot_surface(
        X, Y, Z2, alpha=0.5, label=f"{A[1,0]}x + {A[1,1]}y + {A[1,2]}z = {b[1]}"
    )  # Agregamos la ecuación
    ax.plot_surface(
        X, Y, Z3, alpha=0.5, label=f"{A[2,0]}x + {A[2,1]}y + {A[2,2]}z = {b[2]}"
    )  # Agregamos la ecuación

    plt.xlabel("X")
    plt.ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")  # Colocamos la leyenda en la esquina superior derecha
    plt.show()


# Ejemplo de uso 1
A1 = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]])
b1 = np.array([8, -11, -3])
graficar_solucion(A1, b1)

# # Ejemplo de uso 2
# A2 = np.array([[1, -1, 1], [2, 1, -1], [3, -2, 2]])
# b2 = np.array([2, 1, 5])
# graficar_solucion(A2, b2)

# # Ejemplo de uso 3
# A3 = np.array([[3, 1, -2], [2, -2, 4], [1, 1, 3]])
# b3 = np.array([7, 6, 13])
# graficar_solucion(A3, b3)
