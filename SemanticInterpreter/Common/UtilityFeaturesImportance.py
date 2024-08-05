# Databricks notebook source
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

def pipelineOneVsAllClassifier(
    inputCol,features
):
    # Create the stages
    stringIndexer = StringIndexer(inputCol=inputCol, outputCol="label")
    assembler = VectorAssembler(inputCols=features, outputCol="features_col", handleInvalid="keep")
    rfClassifier = RandomForestClassifier(labelCol="label", featuresCol="features_col", maxBins=256)

    # Create the pipeline with the correct stages
    pipeline = Pipeline(stages=[stringIndexer, assembler, rfClassifier])

    return pipeline

# COMMAND ----------

def splitBilancedDataset(df,inputCol):
    df=df.withColumnRenamed("prediction", "old_prediction")
    # Dividi il DataFrame in base al valore target
    df_0 = df.filter(col(inputCol) == 0)
    df_1 = df.filter(col(inputCol) == 1)

    #  Esegui lo split per ciascun DataFrame
    train_0, test_0 = df_0.randomSplit([0.7, 0.3])
    train_1, test_1 = df_1.randomSplit([0.7, 0.3])

    # Unisci i set di training e test
    train_data = train_0.union(train_1)#.select("Cluster_0","features_col","label")
    test_data = test_0.union(test_1)#.select("Cluster_0","features_col","label")

    return train_data, test_data

# COMMAND ----------

def featuresImportance(model_rf,feature_names):
    # Get feature importance from the trained model
    importance = model_rf.stages[-1].featureImportances.toArray()

    #  Create a dictionary to store feature importance values
    feature_importance_dict = {}
    for i in range(len(feature_names)):
        feature_importance_dict[feature_names[i]] = importance[i]

    # Sort the dictionary by feature importance in descending order
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted feature importance
    #for feature, importance in sorted_feature_importance:
    #    print(feature, ":", importance)
    return sorted_feature_importance

# COMMAND ----------

def featuresImportanceCluster(df, features, file_path_name):
    distinct_predictions = df.select("prediction").distinct().orderBy("prediction").rdd.flatMap(lambda x: x).collect()
    risultati = {}
    i=0

    for pred_val in distinct_predictions:
        print("Valore di predizione: {}".format(pred_val))
        df=df.withColumn("Cluster_"+str(pred_val),when(col("prediction")==pred_val,1).otherwise(0))

        train_data, test_data=splitBilancedDataset(df,"Cluster_"+str(pred_val))
        model_rf=pipelineOneVsAllClassifier("Cluster_"+str(pred_val),features).fit(train_data)

        evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
        accuracy = evaluator.evaluate(model_rf.transform(test_data))

        fi=featuresImportance(model_rf,features)

        risultati[i] = {
            "id_cluster": pred_val,
            "features_importance": fi,
            "accuracy": accuracy
        }
        i=i+1

    file_path = file_path_name
    result_json = json.dumps(risultati)

    with open(file_path, "w") as f:
        f.write(result_json)
