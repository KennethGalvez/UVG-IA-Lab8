# UVG-IA-Lab8

# PCA

PCA (Principal Component Analysis) es una técnica de análisis de datos que se utiliza para reducir la dimensión de un conjunto de datos mediante la extracción de características importantes y la eliminación de características redundantes o irrelevantes.

La técnica de PCA se utiliza comúnmente en el análisis de datos y la minería de datos para reducir la complejidad del modelo y mejorar la precisión del análisis. PCA se basa en la idea de que las variables originales están altamente correlacionadas entre sí, y que es posible expresarlas en términos de unas pocas variables independientes, llamadas componentes principales.

Al reducir la dimensionalidad de los datos mediante PCA, se eliminan las características que no son relevantes o que pueden causar ruido en los datos. Como resultado, se puede mejorar la calidad de los clusters obtenidos mediante técnicas de clustering, como K-Means.

# Criterios para decidir entre modelos

depende de los objetivos y características de los datos. Los modelos de mezcla son más apropiados cuando los datos provienen de diferentes distribuciones, contienen outliers, se necesita modelar la probabilidad de pertenencia a un cluster o se desea modelar correlaciones entre variables. Aunque los modelos de mezcla son más flexibles, también pueden ser más complejos computacionalmente y requerir ajuste de parámetros.

# Comparacion Modelos

Luego de haber realizado los dos modelos, entendido sus funcionamiento y comparado. (lab 7 y 8) hallamos que existe una gran diferencia en cuanto a performance de los modelos. Pues kmeans tarda notablemente menos tiempo que mixture model. Esto se debe uno a la cantidad de datos que manejamos en los dos sistemas y dos, a la cantidad de procesos que se deben realizar. Implementamos el sistema de PCA para reducir la cantidad de variables, sin embargo el sistema sigue siendo muy pesado para mixture models. Con respecto al accuracy o acertacion de los modelos, encontramos que es bastante similar, sin embargo fue más facil analizarlo en el caso de k-means porque es menos complejo.
