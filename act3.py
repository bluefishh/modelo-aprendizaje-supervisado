
# Modelo supervisado para predecir estaciones terminales en Transmilenio

import pandas as pd
import networkx as nx
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Se carga el archivo CSV con los datos de las estaciones
datos = pd.read_csv("Estaciones_Troncales_de_TRANSMILENIO.csv")

# Se crea el grafo
G = nx.Graph()

# Se añaden nodos con atributos de coordenadas y trazado, usando ID y nombre para mejor identificación
for _, fila in datos.iterrows():
    id_nombre = f"{fila['cod_nodo']} - {fila['nom_est']}"
    G.add_node(id_nombre, x=fila["X"], y=fila["Y"], trazado=fila["id_trazado"])

# Se añaden aristas entre estaciones consecutivas en cada trazado, con peso basado en la distancia euclidiana (distancia de una recta entre dos puntos, usando
# teorema de Pitágoras)
for trazado, grupo in datos.groupby("id_trazado"):
    grupo = grupo.sort_values(["X", "Y"])
    for i in range(len(grupo) - 1):
        a = f"{grupo.iloc[i]['cod_nodo']} - {grupo.iloc[i]['nom_est']}"
        b = f"{grupo.iloc[i+1]['cod_nodo']} - {grupo.iloc[i+1]['nom_est']}"
        distancia = math.hypot(grupo.iloc[i]["X"] - grupo.iloc[i+1]["X"],
                               grupo.iloc[i]["Y"] - grupo.iloc[i+1]["Y"])
        G.add_edge(a, b, weight=distancia)  

# Se calculan métricas de centralidad, es decir características de cada nodo en el grafo
grado = dict(G.degree())
cercania = nx.closeness_centrality(G, distance="weight")
intermediacion = nx.betweenness_centrality(G, weight="weight")

# Se crea un DataFrame con las características
caracteristicas = pd.DataFrame({
    "id_nombre": list(G.nodes()),
    "grado": [grado[n] for n in G.nodes()],
    "cercania": [cercania[n] for n in G.nodes()],
    "intermediacion": [intermediacion[n] for n in G.nodes()],
})

# Se define la variable objetivo (etiqueta) basada en el grado del nodo (grado 1 indica estación terminal, grado > 1 indica estación intermedia)
caracteristicas["es_terminal"] = (caracteristicas["grado"] == 1).astype(int)

# Se prepara el conjunto de datos para el modelo supervisado
X = caracteristicas[["cercania", "intermediacion"]]
y = caracteristicas["es_terminal"]

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Se normalizan los datos para mejorar el rendimiento del modelo
escalador = StandardScaler()
X_entrenamiento_s = escalador.fit_transform(X_entrenamiento)
X_prueba_s = escalador.transform(X_prueba)

# Se entrena el modelo de clasificación (Random Forest porque se predice una variable categórica, si fuera numérica se usaría regresión lineal)
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_entrenamiento_s, y_entrenamiento)
y_pred = modelo.predict(X_prueba_s)

# Se hacen predicciones para todas las estaciones y se comparan con las etiquetas reales
caracteristicas["prediccion"] = modelo.predict(escalador.transform(X))
caracteristicas["resultado"] = caracteristicas.apply(
    lambda f: "Correcto" if f["es_terminal"] == f["prediccion"] else "Error",
    axis=1
)

# Se guardan los resultados en un archivo CSV con la validación de todas las estaciones
caracteristicas.to_csv("prediccion_estaciones.csv", index=False, encoding="utf-8-sig")
print("\nArchivo 'prediccion_estaciones.csv' guardado con todas las estaciones.")

# Se grafica el grafo con las predicciones, usando colores para indicar estaciones terminales correctamente detectadas, no detectadas, 
# falsos positivos y estaciones normales
posiciones = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}

colores_nodos = []
for n in G.nodes():
    real = caracteristicas.loc[caracteristicas["id_nombre"] == n, "es_terminal"].values[0]
    pred = caracteristicas.loc[caracteristicas["id_nombre"] == n, "prediccion"].values[0]
    if real == 1 and pred == 1:
        colores_nodos.append("green")
    elif real == 1 and pred == 0:
        colores_nodos.append("red")
    elif real == 0 and pred == 1:
        colores_nodos.append("orange")
    else:
        colores_nodos.append("lightgray")

plt.figure(figsize=(12, 9))
nx.draw(
    G, posiciones,
    with_labels=True,
    labels={n: n for n in G.nodes()},
    node_color=colores_nodos,
    node_size=150,
    font_size=6,
    edge_color="gray",
    alpha=0.8
)

plt.title("Predicción de estaciones terminales (ID - Nombre de estación)", fontsize=12)
plt.scatter([], [], c="green", label="Terminal correcta")
plt.scatter([], [], c="red", label="Terminal no detectada")
plt.scatter([], [], c="orange", label="Falso positivo")
plt.scatter([], [], c="lightgray", label="Estación normal")
plt.legend(scatterpoints=1, frameon=False, loc="upper right", fontsize=8)
plt.show()
