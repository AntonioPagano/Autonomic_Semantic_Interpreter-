# Databricks notebook source
# MAGIC %run /CO/Common/Common_Functions

# COMMAND ----------

# MAGIC %run /Repos/pagan133@posteitaliane.it/DTO_CustomerOperation/Spark_Databricks/ML_Project/Clustering/Common/UtilityClustering

# COMMAND ----------

# MAGIC %run /Repos/pagan133@posteitaliane.it/DTO_CustomerOperation/Spark_Databricks/ML_Project/Clustering/Common/UtilityNLP

# COMMAND ----------

df=spark.read.load('abfss://tables@palignedscrivadl01azwe.dfs.core.windows.net/DP_Check_Scrivania_PDB_CURATED').select(col("`boapclpratica.idcase`").alias("idcase"),col("`boapclanomalie.descrizione`").alias("descrizione"),col("`boapclanomalie.statoanomalia`").alias("statoanomalia")).where("descrizione!='NULL' and descrizione!='' and descrizione!='Verifica Documenti'").distinct()
df.cache()
df.display()

# COMMAND ----------

model = get_nlp_spark_pipeline("descrizione").fit(df)
result_bert = model.transform(df)

# COMMAND ----------

result_bert.display()

# COMMAND ----------

print("# dataset originale: {}".format(df.count()))
print("# dataset nuovo: {}".format(result_bert.count()))

# COMMAND ----------

from pyspark.sql.functions import *

result_df = result_bert.select(
    "idcase",
    "descrizione",
    explode(
        arrays_zip(
            result_bert.sentence.result, result_bert.sentence_bert_embeddings.embeddings
        )
    ).alias("cols")
).select(
    "idcase",
    "descrizione",
    col("cols.result").alias("sentence"),
    col("cols.embeddings").alias("Bert_sentence_embeddings")
)
result_df.display()

# COMMAND ----------

result_df_exp = result_df.select([col("idcase")] + [col("descrizione")] + [col("sentence")] + [col("Bert_sentence_embeddings")[i] for i in range(768)])

col_features=result_df_exp.columns[3:]
result_df_exp_filled = result_df_exp.dropna()
result, pca_model, loadings=pipelineStandardPCA(result_df_exp_filled, col_features, 30)

# COMMAND ----------

cumulativePCwithVariance(pca_model)

# COMMAND ----------

silhouetteClusteringKMeans(result,"pca_features",m=2,n=20,i=2)

# COMMAND ----------

predictions_cluster_final, final_model=defineClustering(result, 12)

# COMMAND ----------

predictions_cluster_final.select("idcase",
    "descrizione",
    "sentence","prediction").distinct().display()

# COMMAND ----------

plotPCA3DInterattivo(predictions_cluster_final.where("idcase like '8%'"), features='pca_features', predictions='prediction', additional_column='descrizione')

# COMMAND ----------

plotClustering3DInterattivo(predictions_cluster_final.where("idcase like '8%'"), features='pca_features', predictions='prediction', additional_column='descrizione')

# COMMAND ----------

from pyspark.ml.feature import NGram
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, FloatType

docs_per_topic = predictions_cluster_final.groupby('prediction').agg(concat_ws(' ', collect_list(col("sentence"))).alias('Doc'))

# definiamo la funzione c_tf_idf:
def c_tf_idf(df, inputCol="Doc", outputCol="features", ngram=2):
    tokenizer = Tokenizer(inputCol=inputCol, outputCol="words")
    ngram = NGram(n=ngram, inputCol="words", outputCol="ngrams")
    cv = CountVectorizer(inputCol="ngrams", outputCol="raw_features", vocabSize=10000, minDF=2, maxDF=df.count()/2)
    idf = IDF(inputCol="raw_features", outputCol=outputCol)
    
    pipeline = Pipeline(stages=[tokenizer, ngram, cv, idf])
    model = pipeline.fit(df)
    result_tfidf = model.transform(df)
    
    return result_tfidf, model

tf_idf, model = c_tf_idf(docs_per_topic,ngram=3)

# Definisci la funzione udf
def extract_indices_values(vector):
    return [(int(index), float(value)) for index, value in zip(vector.indices, vector.values)]

# Definisci lo schema per il tuo nuovo DataFrame
schema = ArrayType(StructType([
    StructField("index", IntegerType(), nullable=False),
    StructField("value", FloatType(), nullable=False)
]))

vocab = model.stages[2].vocabulary

# Registra la tua funzione udf con lo schema
extract_indices_values_udf = udf(extract_indices_values, schema)

# Applica la funzione udf alla colonna 'features'
df_with_array = tf_idf.withColumn("features_array", extract_indices_values_udf(tf_idf["features"]))

# Ora puoi utilizzare explode sulla colonna 'features_array'
#exploded = df_with_array.select('prediction', explode('features_array').alias('feature'))

# Mappa l'indice della parola al suo valore nel vocabolario
mapped = df_with_array.select('prediction', 'words', 'features_array')\
    .rdd.map(lambda row: (row['prediction'], [(vocab[i[1][0]], i[1][1]) for i in enumerate(row['features_array'])]))\
    .toDF(['prediction', 'words'])

# Espandi il DataFrame in modo che ogni riga contenga una singola parola con il suo peso TF-IDF
exploded = mapped.select('prediction', explode('words').alias('word_tfidf'))

# Dividi la colonna 'word_tfidf' in due colonne 'word' e 'tfidf'
exploded = exploded.withColumn('word', col('word_tfidf').getItem('_1'))
exploded = exploded.withColumn('tfidf', col('word_tfidf').getItem('_2'))

# Seleziona solo le colonne 'prediction', 'word' e 'tfidf'
exploded = exploded.select('prediction', 'word', 'tfidf')
N=10
# Ordina i risultati per peso TF-IDF e seleziona le prime N parole per ciascuna prediction
topN = exploded.orderBy('tfidf', ascending=False).groupBy('prediction').agg(collect_list('word').alias('words'))\
    .rdd.map(lambda row: (row['prediction'], row['words'][:N]))\
    .toDF(['prediction', 'topN_words'])

# COMMAND ----------

topN.display()

# COMMAND ----------

predictions_cluster_final.groupBy("prediction").count().display()

# COMMAND ----------


