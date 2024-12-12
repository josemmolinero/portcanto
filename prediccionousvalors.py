"""
@ IOC - CEI_AB_M03_EAC6_2425S1 - JOSE MIGUEL GARCIA MOLINERO
"""

import pickle
from clustersciclistes import nova_prediccio

# Nuevos ciclistas a clasificar
nous_ciclistes = [
    [500, 3230, 1430, 4660],  # BEBB
    [501, 3300, 2120, 5420],  # BEMB
    [502, 4010, 1510, 5520],  # MEBB
    [503, 4350, 2200, 6550]   # MEMB
]

# Cargar el modelo entrenado
with open('model/clustering_model.pkl', 'rb') as f:
    clustering_model = pickle.load(f)

# Usar la función `nova_prediccio` para clasificar
df_nous_ciclistes, pred = nova_prediccio(nous_ciclistes, clustering_model)

# Asociar los clusters con los tipos
tipus = [{'name': 'BEBB', 'label': 0}, {'name': 'BEMB', 'label': 1},
         {'name': 'MEBB', 'label': 2}, {'name': 'MEMB', 'label': 3}]

# Mostrar los resultados
for i, p in enumerate(pred):
    tipo = [t['name'] for t in tipus if t['label'] == p][0]
    print(f"Ciclista {df_nous_ciclistes.iloc[i]['id']} asignado al tipo: {tipo}")
