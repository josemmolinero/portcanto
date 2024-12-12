"""
@ IOC - CEI_AB_M03_EAC6_2425S1 - JOSE MIGUEL GARCIA MOLINERO
"""

import os
import sys
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score

# Configuración básica de logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# Contexto para suprimir mensajes innecesarios
@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w', encoding='utf-8') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


# Carga del dataset
def load_dataset(path):
    """
    Carga un archivo CSV como un DataFrame de Pandas.
    
    Args:
        path (str): Ruta al archivo CSV.
    
    Returns:
        pd.DataFrame: DataFrame cargado.
    """
    try:
        df = pd.read_csv(path, delimiter=';')
        logging.info("Dataset cargado correctamente desde %s", path)
        return df
    except FileNotFoundError:
        logging.error("No se encuentra el archivo en la ruta especificada: %s", path)
        return None


# Análisis exploratorio de datos (EDA)
def realizar_eda(df):
    """
    Realiza un análisis exploratorio de datos e imprime información básica del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar.
    """
    logging.info("Shape del DataFrame: %s", df.shape)
    logging.info("Primeras filas:\n%s", df.head())
    logging.info("Columnas:\n%s", df.columns)
    df.info()


# Limpieza del DataFrame
def clean(df):
    """
    Elimina las columnas no necesarias del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original.
    
    Returns:
        pd.DataFrame: DataFrame limpio.
    """
    columnas_a_eliminar = ['id', 'tt']
    df_clean = df.drop(columns=columnas_a_eliminar, axis=1)
    logging.info("Columnas eliminadas: %s", columnas_a_eliminar)
    return df_clean


# Extracción de etiquetas verdaderas
def extract_true_labels(df):
    """
    Extrae las etiquetas de tipo (BEBB, ...) del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame original.
    
    Returns:
        np.ndarray: Arreglo con las etiquetas verdaderas.
    """
    true_labels_array = df['tipus'].to_numpy()
    logging.info("Etiquetas extraídas: %s", set(true_labels_array))
    return true_labels_array


# Visualización de pairplot
def visualitzar_pairplot(df):
    """
    Genera y guarda un pairplot del DataFrame para analizar relaciones entre atributos.
    
    Args:
        df (pd.DataFrame): DataFrame con los atributos.
    """
    sns.pairplot(df)
    img_path = "img/pairplot.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path)
    logging.info("Pairplot guardado en %s", img_path)


# Clustering con KMeans
def clustering_kmeans(data, n_clusters=4):
    """
    Crea y entrena un modelo KMeans con los datos proporcionados.
    
    Args:
        data (pd.DataFrame): Datos para el clustering.
        n_clusters (int): Número de clusters a generar.
    
    Returns:
        KMeans: Modelo entrenado.
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    with suppress_stdout_stderr():
        model.fit(data)
    logging.info("KMeans entrenado con %s", n_clusters)
    return model


# Visualización de clusters
def visualitzar_clusters(data, labels):
    """
    Visualiza los clusters con diferentes colores en un gráfico 2D.
    
    Args:
        data (pd.DataFrame): Datos del clustering.
        labels (np.ndarray): Etiquetas asignadas por el modelo.
    """
    img_path = "img/clusters.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    sns.scatterplot(x='tp', y='tb', data=data, hue=labels, palette="rainbow")
    plt.savefig(img_path)
    logging.info("Gráfico de clusters guardado en %s", img_path)


# Asignación de clusters a patrones
def associar_clusters_patrons(tipus, model):
    """
    Asocia clusters a los patrones predefinidos (BEBB, BEMB, etc.).
    
    Args:
        tipus (list): Lista de diccionarios con los patrones.
        model (KMeans): Modelo entrenado.
    
    Returns:
        list: Tipos actualizados con los labels asignados.
    """
    dicc = {'tp': 0, 'tb': 1}
    logging.info("Centros de los clusters:")
    for j, center in enumerate(model.cluster_centers_):
        logging.info("Cluster %d: tp=%.2f, tb=%.2f", j, center[dicc['tp']], center[dicc['tb']])

    # Lógica de asignación
    # Procés d'assignació
    ind_label_0 = -1
    ind_label_1 = -1
    ind_label_2 = -1
    ind_label_3 = -1

    suma_max = 0
    suma_min = 50000

    if clustering_model is None:
        logging.error("El modelo de clustering no se pudo entrenar.")
        sys.exit(1)

    for j, center in enumerate(clustering_model.cluster_centers_):
        suma = round(center[dicc['tp']], 1) + round(center[dicc['tb']], 1)
        if suma_max < suma:
            suma_max = suma
            ind_label_3 = j
        if suma_min > suma:
            suma_min = suma
            ind_label_0 = j

    tipus[0].update({'label': ind_label_0})
    tipus[3].update({'label': ind_label_3})

    lst = [0, 1, 2, 3]
    lst.remove(ind_label_0)
    lst.remove(ind_label_3)

    if clustering_model.cluster_centers_[lst[0]][0] < clustering_model.cluster_centers_[lst[1]][0]:
        ind_label_1 = lst[0]
        ind_label_2 = lst[1]
    else:
        ind_label_1 = lst[1]
        ind_label_2 = lst[0]

    tipus[1].update({'label': ind_label_1})
    tipus[2].update({'label': ind_label_2})

    logging.info('\nHem fet l\'associació')
    logging.info('\nTipus i labels:\n%s', tipus)
    return tipus


# Clasificación de nuevos datos
def nova_prediccio(dades, model):
    """
    Predice los clusters para nuevos datos proporcionados.
    
    Args:
        dades (list): Lista de listas con datos de nuevos ciclistas.
        model (KMeans): Modelo de clustering.
    
    Returns:
        tuple: (DataFrame con los datos agrupados, predicciones del modelo).
    """
    df_nuevos = pd.DataFrame(dades, columns=['id', 'tp', 'tb', 'tt'])
    df_clean = clean(df_nuevos)
    predicciones = model.predict(df_clean)
    logging.info("Nuevas predicciones: %s", predicciones)
    return df_nuevos, predicciones

def generar_informes(df, tipus):
    """
    Genera informes en la carpeta 'informes/' a partir del dataset de ciclistas agrupados en
    clusters.
    Se crean 4 archivos, uno por cada cluster, con los IDs de los ciclistas asignados a ese
    cluster.

    Args:
        df (pd.DataFrame): DataFrame con los datos de los ciclistas y sus labels.
        tipus (list): Lista de diccionarios que asocia los patrones con los labels de los
        clusters.

    Returns:
        None
    """
    # Crear la carpeta de informes si no existe
    informes_dir = "informes/"
    os.makedirs(informes_dir, exist_ok=True)

    # Generar informes por cada cluster
    for tip in tipus:
        cluster_label = tip['label']
        cluster_name = tip['name']
        file_path = os.path.join(informes_dir, f"{cluster_name}.txt")

        # Filtrar ciclistas del cluster actual
        cluster_data = df[df['label'] == cluster_label]

        # Guardar IDs en el archivo
        with open(file_path, 'w', encoding='utf-8') as file:
            for _, row in cluster_data.iterrows():
                file.write(f"{row['id']}\n")

        logging.info("Informe generado para el cluster '%s' en '%s'", cluster_name, file_path)

    logging.info("Todos los informes han sido generados en la carpeta 'informes/'.")

if __name__ == "__main__":

    # Asegurarse de que los directorios necesarios existen
    os.makedirs('./model', exist_ok=True)
    os.makedirs('./data', exist_ok=True)

    # Cargar el dataset
    PATH_DATASET = './data/ciclistes.csv'
    ciclistes_data = load_dataset(PATH_DATASET)

    if ciclistes_data is None or ciclistes_data.empty:
        logging.error("El dataset no se pudo cargar o está vacío.")
        sys.exit(1)

    # Análisis exploratorio
    realizar_eda(ciclistes_data)

    # Limpiar datos y extraer etiquetas verdaderas
    ciclistes_data_clean = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data_clean)

    # Eliminar la columna 'tipus', ya no necesaria para clustering
    if 'tipus' in ciclistes_data_clean.columns:
        ciclistes_data_clean = ciclistes_data_clean.drop('tipus', axis=1)

    # Visualización inicial de los datos
    visualitzar_pairplot(ciclistes_data_clean)

    # Clustering con KMeans
    logging.info('\nDades per l\'entrenament:\n%s\n...', ciclistes_data_clean[:3])
    clustering_model = clustering_kmeans(ciclistes_data_clean, 4)

    # Guardar el modelo entrenado
    with open('model/clustering_model.pkl', 'wb') as f:
        pickle.dump(clustering_model, f)

    # Etiquetas generadas por el modelo
    data_labels = clustering_model.labels_

    # Calcular y registrar métricas de evaluación
    homogeneity = homogeneity_score(true_labels, data_labels)
    completeness = completeness_score(true_labels, data_labels)
    v_measure = v_measure_score(true_labels, data_labels)

    logging.info('\nHomogeneity: %.3f', homogeneity)
    logging.info('Completeness: %.3f', completeness)
    logging.info('V-measure: %.3f', v_measure)

    # Guardar métricas de evaluación
    with open('model/scores.pkl', 'wb') as f:
        pickle.dump({
            "h": homogeneity,
            "c": completeness,
            "v": v_measure
        }, f)

    # Visualización de los clusters
    visualitzar_clusters(ciclistes_data, data_labels)

    # Array de diccionarios que asigna los tipos a los labels
    tipus = [{'name': 'BEBB'}, {'name': 'BEMB'}, {'name': 'MEBB'}, {'name': 'MEMB'}]

    # Agregar columna de etiquetas generadas al DataFrame original
    ciclistes_data['label'] = data_labels.tolist()
    logging.info('\nColumna label añadida:\n%s', ciclistes_data[:5])

    # Asociar clusters con patrones específicos
    tipus = associar_clusters_patrons(tipus, clustering_model)

    # Guardar la asociación de tipos
    with open('model/tipus_dict.pkl', 'wb') as f:
        pickle.dump(tipus, f)

    # Generar informes
    generar_informes(ciclistes_data, tipus)

    # Clasificación de nuevos valores
    nous_ciclistes = [
        [500, 3230, 1430, 4670],  # BEBB
        [501, 3300, 2120, 5420],  # BEMB
        [502, 4010, 1510, 5520],  # MEBB
        [503, 4350, 2200, 6550]   # MEMB
    ]

    logging.info('Nous valors:\n%s\n', nous_ciclistes)
    df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)
    logging.info('Predicció dels valors:\n%s\n', pred)

    # Asignar los nuevos valores a los tipos
    for i, p in enumerate(pred):
        t = [t for t in tipus if t['label'] == p]
        logging.info('Tipus %s (%s) - Classe %s', df_nous_ciclistes.index[i], t[0]['name'], p)
