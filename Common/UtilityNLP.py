# Databricks notebook source
def nlp_pipeline_bert_sentence_embedding(text_col):
    """
    Crea un pipeline per l'elaborazione del linguaggio naturale (NLP) che utilizza il modello BERT per l'incorporamento delle frasi.

    Argomenti:
    text_col (str): Il nome della colonna che contiene il testo da elaborare.

    Restituisce:
    pipeline: Un oggetto Pipeline che esegue le seguenti operazioni:
        1. Assembla il testo in un documento.
        2. Esegue la divisione in frasi utilizzando un modello di rilevamento delle frasi.
        3. Tokenizza le frasi in parole.
        4. Normalizza le parole.
        5. Rimuove le stop words italiane.
        6. Applica lo stemming alle parole.
        7. Assembla il documento e le parole stemmate in una colonna di frasi.
        8. Calcola gli embedding BERT delle frasi utilizzando il modello preaddestrato in italiano.
        9. Calcola gli embedding delle frasi utilizzando la strategia di pooling "AVERAGE".

    Esempio:
    pipeline = nlp_pipeline_bert_sentence_embedding("testo")
    model = pipeline.fit(df)
    result = model.transform(df)
    """

    from sparknlp.base import DocumentAssembler
    from sparknlp.annotator import (
        SentenceDetector,
        BertSentenceEmbeddings,
        DistilBertEmbeddings,
        WordEmbeddingsModel,
        Tokenizer,
        StopWordsCleaner,
        Stemmer,
        NGramGenerator,
        SentenceDetectorDLModel,
        Normalizer,
        SentenceEmbeddings,
    )

    from sparknlp.base import TokenAssembler
    from pyspark.ml import Pipeline

    documentAssembler = (
        DocumentAssembler().setInputCol(text_col).setOutputCol("document")
    )

    sentencerDL = (
        SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
        .setInputCols(["document"])
        .setOutputCol("sentences")
    )

    tokenizer = Tokenizer().setInputCols("sentences").setOutputCol("token")

    normalizer = Normalizer().setInputCols(["token"]).setOutputCol("normalized").setCleanupPatterns(["""[^\w\s]|http\S+|\d+|[A-Z][a-z]+"""]).setLowercase(True)

    stop_words = (
        StopWordsCleaner.pretrained("stopwords_it", "it")
        .setInputCols(["normalized"])
        .setOutputCol("rmStopWordTokens")
    )

    stemmer = Stemmer().setInputCols(["rmStopWordTokens"]).setOutputCol("stemWord")

    tokenAssembler = (
        TokenAssembler().setInputCols(["document", "stemWord"]).setOutputCol("sentence")
    )

    embeddings = (
        DistilBertEmbeddings.pretrained("distilbert_embeddings_BERTino", "it")
        .setInputCols(["sentence", "stemWord"])
        .setOutputCol("embeddings")
    )

    embeddingsSentence = (
        SentenceEmbeddings()
        .setInputCols(["sentence", "embeddings"])
        .setOutputCol("sentence_bert_embeddings")
        .setPoolingStrategy("AVERAGE")
    )

    pipeline = Pipeline(
        stages=[
            documentAssembler,
            sentencerDL,
            tokenizer,
            normalizer,
            stop_words,
            stemmer,
            tokenAssembler,
            embeddings,
            embeddingsSentence,
        ]
    )

    return pipeline

# COMMAND ----------

def convert_sentence_embedding_in_col(df, cols=[], result_bert="result_bert"):
    """
    Converte gli embedding delle frasi in colonne separate nel DataFrame.

    Argomenti:
    df (DataFrame): Il DataFrame di input contenente gli embedding delle frasi.
    cols (list, opzionale): Una lista delle colonne da selezionare dal DataFrame di input. Default è una lista vuota.
    result_bert (str, opzionale): Il pattern del nome della colonna risultante che conterrà gli embedding delle frasi. Default è "result_bert".

    Restituisce:
    DataFrame: Un nuovo DataFrame con le colonne selezionate dal DataFrame di input e le colonne "result_bert[0:N]" contenente gli embedding delle frasi, dove N è la dimensione del vettore di embedding.

    Esempio:
    result_df_exp = convert_sentence_embedding_in_col(df, ["col1", "col2"], "result_bert")
    """

    result_df = df.select(
        *cols,
        explode(
            arrays_zip(
                col("sentence.result"),
                col("sentence_bert_embeddings.embeddings"),
            )
        ).alias("cols"),
    ).select(
        *cols,
        col("cols.result").alias("sentence"),
        col("cols.embeddings").alias(result_bert),
    )

    length = result_df.select(size(result_bert).alias("length")).first().length
    result_df_exp = result_df.selectExpr(
        *cols,
        "sentence",
        *[
            f"{result_bert}[{i}] as {result_bert}_{i}"
            for i in range(length)
        ],
    )

    return result_df_exp

# COMMAND ----------

def top_n_words(df, inputCol="Doc", outputCol="features", ngram=2, N=10, targetCol="prediction",minDF=2,perc_max=0.9):
    """
    Restituisce le prime N parole con il peso TF-IDF più alto per ciascuna predizione nel DataFrame di input.

    Argomenti:
    df (DataFrame): Il DataFrame di input contenente i documenti.
    inputCol (str, opzionale): Il nome della colonna che contiene i documenti. Default è "Doc".
    outputCol (str, opzionale): Il nome della colonna risultante che conterrà le features TF-IDF. Default è "features".
    ngram (int, opzionale): La dimensione degli n-grammi da considerare. Default è 2.
    N (int, opzionale): Il numero di parole da selezionare per ciascuna predizione. Default è 10.
    targetCol (str, opzionale): Il nome della colonna che contiene le predizioni. Default è "prediction".
    minDF (int, opzionale): Il numero minimo di documenti in cui un n-gramma deve essere presente per essere considerata. Default è 2.
    perc_max (float, opzionale): La percentuale massima di documenti in cui un n-gramma deve essere presente per essere considerata. Default è 0.9.

    Restituisce:
    DataFrame: Un nuovo DataFrame contenente le predizioni e le prime N parole con il peso TF-IDF più alto per ciascuna predizione.

    Esempio:
    top_words_df = top_n_words(df, inputCol="Doc", outputCol="features", ngram=2, N=10, targetCol="prediction",minDF=2,perc_max=0.9)
    """
    
    from pyspark.ml.feature import NGram
    from pyspark.ml.feature import CountVectorizer, IDF
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Tokenizer
    from pyspark.sql.functions import col, explode, collect_list
    from pyspark.sql.types import ArrayType, StructType, StructField, IntegerType, FloatType
    from pyspark.sql.functions import udf

    def c_tf_idf(df, inputCol=inputCol, outputCol=outputCol, ngram=ngram,minDF=minDF,perc_max=perc_max):
        """
        Calcola il peso TF-IDF per i documenti nel DataFrame di input.

        Argomenti:
        df (DataFrame): Il DataFrame di input contenente i documenti.
        inputCol (str, opzionale): Il nome della colonna che contiene i documenti. Default è "Doc".
        outputCol (str, opzionale): Il nome della colonna risultante che conterrà le features TF-IDF. Default è "features".
        ngram (int, opzionale): La dimensione degli n-grammi da considerare. Default è 2.

        Restituisce:
        DataFrame: Un nuovo DataFrame contenente i documenti e le features TF-IDF calcolate.

        Esempio:
        tfidf_df = c_tf_idf(df, inputCol="Doc", outputCol="features", ngram=2)
        """

        tokenizer = Tokenizer(inputCol=inputCol, outputCol="words")
        ngram = NGram(n=ngram, inputCol="words", outputCol="ngrams")
        cv = CountVectorizer(inputCol="ngrams", outputCol="raw_features", vocabSize=10000, minDF=minDF, maxDF=df.count()*perc_max)
        idf = IDF(inputCol="raw_features", outputCol=outputCol)

        pipeline = Pipeline(stages=[tokenizer, ngram, cv, idf])
        model = pipeline.fit(df)
        result_tfidf = model.transform(df)

        return result_tfidf, model

    def extract_indices_values(vector):
        return [(int(index), float(value)) for index, value in zip(vector.indices, vector.values)]

    schema = ArrayType(StructType([
        StructField("index", IntegerType(), nullable=False),
        StructField("value", FloatType(), nullable=False)
    ]))

    result_tfidf, model = c_tf_idf(df, inputCol, outputCol, ngram)

    vocab = model.stages[2].vocabulary

    extract_indices_values_udf = udf(extract_indices_values, schema)

    df_with_array = result_tfidf.withColumn("features_array", extract_indices_values_udf(result_tfidf[outputCol]))

    mapped = df_with_array.select(targetCol, 'words', 'features_array')\
        .rdd.map(lambda row: (row[targetCol], [(vocab[i[1][0]], i[1][1]) for i in enumerate(row['features_array'])]))\
        .toDF([targetCol, 'words'])

    exploded = mapped.select(targetCol, explode('words').alias('word_tfidf'))

    exploded = exploded.withColumn('word', col('word_tfidf').getItem('_1'))
    exploded = exploded.withColumn('tfidf', col('word_tfidf').getItem('_2'))

    exploded = exploded.select(targetCol, 'word', 'tfidf')

    topN = exploded.orderBy('tfidf', ascending=False).groupBy(targetCol).agg(collect_list('word').alias('words'))\
        .rdd.map(lambda row: (row[targetCol], row['words'][:N]))\
        .toDF([targetCol, 'topN_words'])

    return topN
