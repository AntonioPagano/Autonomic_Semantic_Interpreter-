# Databricks notebook source
# DBTITLE 1,Import Library
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.dataframe import DataFrame 

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, PCA, VectorAssembler
from pyspark.ml.linalg import Vectors

from pyspark.ml.clustering import KMeans
from tqdm import tqdm
from pyspark.ml.evaluation import ClusteringEvaluator

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.express as px
from pyspark.sql.functions import median, mode
import re


# COMMAND ----------

from pyspark.ml.feature import StringIndexer

def encoded_categorical_cols(df : DataFrame, categorical_cols) -> DataFrame:
    """
    Codifica le colonne categoriche di un DataFrame utilizzando StringIndexer.
    
    Parametri:
    df (DataFrame): Il DataFrame contenente le colonne da codificare.
    categorical_cols (list): Una lista delle colonne categoriche.
    
    Returns:
    DataFrame: Il DataFrame con le colonne categoriche codificate.
    """
    
    for col in categorical_cols:
        indexer = StringIndexer(inputCol=col, outputCol=col+"_encoded", handleInvalid="skip")
        df = indexer.fit(df).transform(df)

    return df


# COMMAND ----------

def corrMatrixShow(df : DataFrame, num_cols: list) -> None : 
    """
    Crea una matrice di correlazione tra le colonne numeriche di un DataFrame e mostra una heatmap.
    
    Parameters:
    df (DataFrame): Il DataFrame contenente le colonne da analizzare.
    num_cols (list): Una lista delle colonne numeriche da tenere in considerazione.
    
    Returns:
    None
    """
    # Crea un VectorAssembler per convertire le colonne in un unico vettore
    assembler = VectorAssembler(inputCols=num_cols, outputCol="features")

    # Trasforma il DataFrame in vettore di features
    df_vector = assembler.transform(df).select("features")

    # Calcola la matrice di correlazione
    matrix = Correlation.corr(df_vector, "features")

    # Stampa la matrice di correlazione
    # Ottieni la matrice di correlazione come un array
    corr_matrix = matrix.collect()[0]["pearson({})".format("features")].toArray()

    # Converti l'array in un DataFrame pandas
    pandas_df = pd.DataFrame(corr_matrix, columns=num_cols, index=num_cols)

    # Crea una heatmap utilizzando seaborn
    plt.figure(figsize=(20, 14))
    sns.heatmap(pandas_df, annot=True, cmap='coolwarm')
    display(plt.show())

# COMMAND ----------

def varianzaTotPCA(values:list) -> int:
    """
    Calcola la varianza totale dei valori nel dato elenco.
    
    Parametri:
    values (list): L'elenco dei valori di cui calcolare la varianza totale.
    
    Returns:
    Int: totale varianza mantenuta
    """
    total_var = 0
    for item in values:
        total_var+=item
    return total_var

def pipelineStandardPCA(df : DataFrame, col_feature : list, k : int):
    """
    Crea una pipeline per l'applicazione della PCA su un DataFrame.
    
    Parametri:
    df (DataFrame): Il DataFrame di input.
    col_feature (list): L'elenco delle colonne da utilizzare come feature.
    k (int): Il numero di componenti principali desiderati.
    
    Returns:
    DataFrame : Il DataFrame trasformato utilizzando la PCA.
    pca_model : modello trainato in cui è applicata la PCA
    loadings : Dataframe pandas che contiene i loadings (pesi delle variabili originali rispetto alle PC)
    """
    
    # trasformazione delle colonne in vettore di features 
    assembler = VectorAssembler(inputCols=col_feature, outputCol="features")
    df_transformed = assembler.transform(df)

    #Standardizzazione
    scaler=StandardScaler(inputCol='features',outputCol="scaledFeatures",
                            withStd=True, withMean=False)

    # Train PCA model
    pca = PCA(k=k, inputCol="scaledFeatures", outputCol="pca_features")

    # Crea una pipeline con gli step di assembler, scaler e pca
    pipeline = Pipeline(stages=[assembler, scaler, pca])

    # Addestra la pipeline sul DataFrame
    model = pipeline.fit(df)

    # Trasforma il DataFrame utilizzando la pipeline addestrata
    result = model.transform(df)


    # Ottieni il modello PCA dalla pipeline
    pca_model = model.stages[-1]

    # Ottieni la varianza spiegata da ogni componente principale
    explained_variance = pca_model.explainedVariance

    print("Varianza spiegata da ogni componente principale: ", explained_variance)
    print("Varianza totale: {}".format(varianzaTotPCA(list(pca_model.explainedVariance))))

    component_names = [f"PC{i+1}" for i in range(pca_model.pc.toArray().shape[1])]

    loadings = pd.DataFrame(
        pca_model.pc.toArray(),  #matrix of loadings
        columns=component_names,  # columns are the principal components
        index=col_feature,  # the rows are the original features
        )

    return result, pca_model, loadings


# COMMAND ----------

def cumulativePCwithVariance(pca_model):
    """
    Visualizza la varianza spiegata cumulativa dai componenti principali.
    
    Parametri:
    pca_model (PCA): Il modello PCA addestrato.
    
    Returns:
    None
    """
    # Calcola la varianza individuale
    individual_variances = list(pca_model.explainedVariance)

    # Calcola la varianza cumulativa
    cumulative_variances = np.cumsum(individual_variances)

    # Crea il grafico delle barre per la varianza individuale
    plt.figure(figsize=(12, 7))
    bar = plt.bar(range(1, len(individual_variances) + 1), individual_variances, alpha=0.6, color='g', label='Varianza spiegata individuale')

    # Crea il grafico a linea per la varianza cumulativa
    line = plt.plot(range(1, len(cumulative_variances) + 1), cumulative_variances, marker='o', linestyle='-', color='r', 
                    label='Varianza spiegata cumulativa')

    # Aggiungi i valori percentuali sopra le barre e i punti
    for i, (bar, cum_val) in enumerate(zip(bar, cumulative_variances)):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{individual_variances[i]*100:.1f}%', 
                 ha='center', va='bottom')
        plt.text(i+1, cum_val, f'{cum_val*100:.1f}%', ha='center', va='bottom')

    # Estetica del grafico
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulativa Varianza spiegata')
    plt.title('Varianza spiegata dai diversi Principal Components')
    plt.xticks(range(1, len(individual_variances) + 1))
    plt.legend(loc='upper left')
    plt.ylim(0, 1.1)  # estendi il limite dell'asse y per ospitare le etichette di testo
    plt.grid(True)
    plt.show()


# COMMAND ----------

def silhouetteClusteringKMeans(df, featuresCol='pca_features', m=2, n=20,i=1) -> list:
    """
    Calcola il coefficiente di silhouette per il clustering K-means su un DataFrame.
    
    Parametri:
    df (DataFrame): Il DataFrame contenente i dati di input.
    featuresCol (str): Nome della colonna feature.
    m (int): Valore minimo di k da testare.
    n (int): Valore massimo di k da testare.
    i (int): Passo incrementale tra i valori di k da testare.
    
    Returns:
    list: Una lista di coefficienti di silhouette per ciascun valore di k testato.
    """
    ks_to_test = range(m,n,i)

    silhouettes = []
    distortions = []

    for k in tqdm(ks_to_test):
        kmeans = KMeans(featuresCol=featuresCol, k=k)  # Inizializza KMeans con un valore di K
        model_cluster = kmeans.fit(df)  # Adatta il modello ai dati
        predictions_cluster = model_cluster.transform(df)  # Applica il modello per ottenere le previsioni

        distortion = model_cluster.summary.trainingCost
        distortions.append(distortion)

        # Calcola il coefficiente di silhouette
        evaluator = ClusteringEvaluator(featuresCol=featuresCol, metricName='silhouette', distanceMeasure='squaredEuclidean')
        silhouette = evaluator.evaluate(predictions_cluster)

        silhouettes.append(silhouette)
    
    # Crea il grafico delle silhouette
    plotSilhouttes(silhouettes, ks_to_test)

    return silhouettes


def plotSilhouttes(silhouettes, ks_to_test):
    """
    Crea un grafico delle silhouettes per valutare i risultati del clustering K-means.
    Parametri:
    silhouettes (list): Lista di valori di silhouette ottenuti nel clustering k-means.
    ks_to_test (list): Lista di passi incrementali per i valori di k da testare

    Returns:
    None
    """
    plt.figure(figsize=(6,6), dpi=200)

    plt.plot(ks_to_test, silhouettes)
    plt.scatter(ks_to_test, silhouettes)

    datetime_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(f'./silhouette_{datetime_now}.png')
    plt.show()


# COMMAND ----------

def defineClustering(df, k, featuresCol='pca_features', maxIter=1000, seed=42):
    """
    Definisce il clustering K-means su un DataFrame con un dato numero di cluster.
    
    Parametri:
    df (DataFrame): Il DataFrame contenente i dati di input.
    k (int): Numero di cluster desiderato.
    featuresCol (str): Nome della colonna feature.
    
    Returns:
    DataFrame: Il DataFrame con le previsioni del clustering.
    KMeansModel: Il modello di clustering K-means adattato ai dati.
    """
    
    kmeans = KMeans(featuresCol=featuresCol,k=k, maxIter=1000, seed=42)  # Inizializza KMeans con un valore di K
    final_model = kmeans.fit(df)  # Adatta il modello ai dati
    predictions_cluster_final = final_model.transform(df)  # Applica il modello per ottenere le previsioni  

    return predictions_cluster_final, final_model


# COMMAND ----------

def plotClustering3D(predictions_cluster_final,features='pca_features',predictions='prediction'):
    """
    Crea un grafico tridimensionale per visualizzare i risultati del clustering K-means.
    
    Parametri:
    predictions_cluster_final (DataFrame): Il DataFrame con le previsioni del clustering.
    features (str): Nome della colonna feature.
    predictions (str): Nome della colonna con le previsioni del clustering.
    
    Returns:
    None
    """
    # Convertire il DataFrame PySpark in un DataFrame Pandas
    pandas_df = predictions_cluster_final.select(features, predictions).toPandas()

    # Estrarre le componenti principali
    pandas_df["pca1"] = pandas_df[features].apply(lambda x: x[0])
    pandas_df["pca2"] = pandas_df[features].apply(lambda x: x[1])
    pandas_df["pca3"] = pandas_df[features].apply(lambda x: x[2])

    # Creare il grafico 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Assegnare colori distinti a ciascun cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, pandas_df[predictions].nunique()))
    for i, color in zip(range(pandas_df[predictions].nunique()), colors):
        cluster = pandas_df[pandas_df[predictions] == i]
        ax.scatter(cluster["pca1"], cluster["pca2"], cluster["pca3"], c=color)

    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    plt.show()

# COMMAND ----------

def plotPCA3DInterattivo(predictions_cluster_final, features='pca_features', predictions='prediction', additional_column=None):
    """
    Crea un grafico 3D interattivo per visualizzare i risultati del clustering K-means.
    
    Parametri:
    predictions_cluster_final (DataFrame): Il DataFrame con le previsioni del clustering.
    features (str): Nome della colonna feature (assi di plot).
    predictions (str): Nome della colonna con le previsioni del clustering.
    additional_column (str): Nome della colonna aggiuntiva da visualizzare al passaggio del cursore.
    
    Returns:
    None
    """
    
    # Convertire il DataFrame PySpark in un DataFrame Pandas
    pandas_df = predictions_cluster_final.select(features, predictions, additional_column).toPandas()

    # Estrarre le componenti principali
    pandas_df["pca1"] = pandas_df[features].apply(lambda x: x[0])
    pandas_df["pca2"] = pandas_df[features].apply(lambda x: x[1])
    pandas_df["pca3"] = pandas_df[features].apply(lambda x: x[2])

    # Crea il grafico 3D interattivo
    fig = px.scatter_3d(pandas_df, x='pca1', y='pca2', z='pca3', hover_data=[additional_column])
    
    # Aggiungi la possibilità di ruotare il grafico
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))))
    
    # Mostra il grafico interattivo
    fig.show()

# COMMAND ----------

def plotClustering3DInterattivo(predictions_cluster_final, features='pca_features', predictions='prediction',additional_column=None):
    """
    Crea un grafico 3D interattivo per visualizzare i risultati del clustering K-means.
    
    Parametri:
    predictions_cluster_final (DataFrame): Il DataFrame con le previsioni del clustering.
    features (str): Nome della colonna feature (assi di plot).
    predictions (str): Nome della colonna con le previsioni del clustering.
    
    Returns:
    None
    """
    
    # Convertire il DataFrame PySpark in un DataFrame Pandas
    pandas_df = predictions_cluster_final.select(features, predictions,additional_column).toPandas()

    # Estrarre le componenti principali
    pandas_df["pca1"] = pandas_df[features].apply(lambda x: x[0])
    pandas_df["pca2"] = pandas_df[features].apply(lambda x: x[1])
    pandas_df["pca3"] = pandas_df[features].apply(lambda x: x[2])

    # Crea il grafico 3D interattivo
    fig = px.scatter_3d(pandas_df, x='pca1', y='pca2', z='pca3',
                        color=predictions,color_discrete_sequence=px.colors.qualitative.Vivid,hover_data=[additional_column],size_max=20)
    
    # Aggiungi la possibilità di ruotare il grafico
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))))
    
    # Mostra il grafico interattivo
    fig.show()


# COMMAND ----------

def remove_pattern(text, pattern):
    """
    Rimuove un pattern da un testo utilizzando le espressioni regolari.
    
    Parametri:
    text (str): Il testo da elaborare.
    pattern (str): Il pattern da rimuovere.
    
    Returns:
    str: Il testo senza il pattern specificato.
    """
    regex = re.compile(pattern)
    return regex.sub("", text)


# COMMAND ----------

def centroidsOriginFeatures(df: DataFrame, col_end: list, predictions='prediction') -> DataFrame:
    """
    Calcola i centroidi dei cluster rispetto alle feature originali.

    Parametri:
    df (DataFrame): Il DataFrame dei dati.
    col_end (list): Una lista di colonne con feature codificate.

    Returns:
    DataFrame: Il DataFrame dei centroidi dei cluster.
    """

    # Calcola i centroidi dei cluster rispetto alle feature originali
    exprs = [mode(col).alias(col) if "_encoded" in col else median(col).alias(col) for col in col_end]
    centroids = df.groupBy(predictions).agg(*exprs)

    for c in col_end:
        # Rimuove il pattern "_encoded" dal nome della colonna
        col_org = remove_pattern(c, "_encoded")
        print(f"Elaborazione colonna {c} -> {col_org}")

        # Seleziona la colonna codificata e la corrispondente colonna originale
        df_tmp = df.select(c, col_org).distinct()

        # Unisci i dati dei centroidi con le colonne originale e codificata per quella feature
        centroids = centroids.join(df_tmp, c, "left")

    return centroids

# COMMAND ----------


