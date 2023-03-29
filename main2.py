import pandas as pd

# Importar las bibliotecas necesarias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Cargar el conjunto de datos
df = pd.read_csv("cleaned_bank_transactions.csv")

# Seleccionar las columnas que se utilizarán en el modelo
X = df

# Estandarizar los datos
scaler = StandardScaler()
data = scaler.fit_transform(X)

# Encontrar el numero de clusters ideal con el método del codo
sse = []
for i in range(1, 11):
    model = GaussianMixture(n_components=i, covariance_type="full", random_state=42)
    model.fit(data)
    sse.append(model.bic(data))
plt.plot(range(1, 11), sse, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("SSE")
plt.show()


# Ajustar el modelo de Mixture Models utilizando el número de clusters seleccionado
model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
model.fit(data)

# Utilizar PCA para visualizar los datos en dos dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)

# Mostrar gráficamente cómo se dividen los clusters seleccionados y mostrar la covarianza como una elipse para cada uno de los clusters
labels = model.predict(data)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
for i in range(model.n_components):
    mean = pca.transform(model.means_[i].reshape(1, -1))
    covar = np.cov(data.T)
    v, w = np.linalg.eigh(covar)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convertir a grados
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean[0], v[0], v[1], 180 + angle, color="black")
    ell.set_clip_box(plt.gca().bbox)
    ell.set_alpha(0.5)
    plt.gca().add_artist(ell)
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.title("Segmentación de Clientes utilizando Mixture Models")
plt.show()

from sklearn.metrics import calinski_harabasz_score

calinski_harabasz_score(data, labels)


# Evaluar el desempeño del modelo utilizando la métrica de Silhouette Score
# silhouette_score(data, labels)
