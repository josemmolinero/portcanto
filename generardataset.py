"""
@ IOC - CEI_AB_M03_EAC6_2425S1 - JOSE MIGUEL GARCIA MOLINERO
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

def generar_dataset(num, initial_id, dicc):
    """
    Genera los tiempos de los ciclistas de forma aleatoria según los parámetros de una categoría.
    
    Args:
        num (int): Número de ciclistas a generar.
        initial_id (int): Identificador inicial de los ciclistas (dorsal).
        dicc (dict): Parámetros de la categoría.
    
    Returns:
        list: Lista de registros de ciclistas.
    """
    datos = []
    for i in range(num):
        dorsal = initial_id + i
        tiempo_p = max(0, int(np.random.normal(dicc['mu_p'], dicc['sigma'])))
        tiempo_b = max(0, int(np.random.normal(dicc['mu_b'], dicc['sigma'])))
        tiempo_total = tiempo_p + tiempo_b
        datos.append({
            "id": dorsal,
            "tp": tiempo_p,
            "tb": tiempo_b,
            "tt": tiempo_total,
            "tipus": dicc['name']
        })
    return datos

if __name__ == "__main__":
    OUTPUT_PATH = 'data/ciclistes.csv'

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Parámetros de las categorías
    # Media tiempo subida buenos escaladores
    MU_P_BE = 3240
    # Media tiempo subida malos escaladores
    MU_P_ME = 4268
    # Media tiempo bajada buenos bajadores
    MU_B_BB = 1440
    # Media tiempo bajada malos bajadores
    MU_B_MB = 2160
    # Desviación estándar
    SIGMA = 240

    categorias = [
        {"name": "BEBB", "mu_p": MU_P_BE, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "BEMB", "mu_p": MU_P_BE, "mu_b": MU_B_MB, "sigma": SIGMA},
        {"name": "MEBB", "mu_p": MU_P_ME, "mu_b": MU_B_BB, "sigma": SIGMA},
        {"name": "MEMB", "mu_p": MU_P_ME, "mu_b": MU_B_MB, "sigma": SIGMA}
    ]

    NUM_CICLISTAS_POR_CATEGORIA = 100
    registros = []
    START_ID = 1

    for categoria in categorias:
        registros.extend(generar_dataset(NUM_CICLISTAS_POR_CATEGORIA, START_ID, categoria))
        START_ID += NUM_CICLISTAS_POR_CATEGORIA

    # Crear un DataFrame con los datos y ordenar por Tiempo Total
    df = pd.DataFrame(registros)
    df.sort_values(by="tt", inplace=True)

    # Guardar el dataset como archivo CSV
    df.to_csv(OUTPUT_PATH, index=False, sep=';')
    logging.info("Archivo generado en: %s", OUTPUT_PATH)
