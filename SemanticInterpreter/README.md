# Autonomic Semantic Interpreter

### Descrizione del Progetto
Questo progetto utilizza PySpark e la libreria Spark NLP per implementare un modello <b/>RAG (Retrieval-Augmented Generation)</b> orientato all'analisi semantica di frasi in italiano. L'obiettivo è identificare argomenti comuni tra le frasi utilizzando tecniche di elaborazione del linguaggio naturale e machine learning. I principali passaggi del flusso di lavoro sono i seguenti:

1. **Sentence Embedding con BERT:** Viene utilizzato un modello BERT pre-addestrato per generare embedding delle frasi in italiano, trasformandole in vettori numerici che catturano il loro significato semantico.

2. **Riduzione della Dimensione con PCA:** Per ottimizzare il clustering e ridurre la complessità computazionale, i vettori delle frasi vengono ridotti dimensionalmente utilizzando l'analisi delle componenti principali (PCA).

3. **Clustering:** I vettori ridotti vengono raggruppati in cluster utilizzando algoritmi di clustering non supervisionati, al fine di identificare frasi simili che condividono concetti semantici.

4. **Analisi del Contenuto con TF-IDF:** Una volta formati i cluster, si utilizza la tecnica TF-IDF per analizzare il contenuto delle frasi in ciascun cluster, estraendo il valore semantico dominante che rappresenta l'argomento comune discusso.

Questo approccio consente di comprendere e organizzare automaticamente le frasi in base ai temi trattati, facilitando l'analisi del linguaggio naturale e l'estrazione di informazioni utili da insiemi di testi non strutturati.


# Setup per l'Esecuzione Locale di PySpark e Spark-NLP

Questo README fornisce le istruzioni passo-passo per configurare un ambiente locale per eseguire il software PySpark e Spark-NLP. Segui questi passaggi per assicurarti che tutto sia configurato correttamente sul tuo sistema Windows.

* PySpark in locale può funzionare senza HADOOP_HOME perché non ha bisogno di un'installazione completa di Hadoop se usato su piccola scala o in modalità standalone.
* Spark NLP, e in generale qualsiasi libreria che utilizza spark.jars.packages, richiede la configurazione di HADOOP_HOME perché si affida alla struttura di Hadoop per gestire le dipendenze, il download dei pacchetti e altre operazioni legate all'esecuzione distribuita.

## Requisiti di Sistema

- **Sistema Operativo:** Windows 10/11
- **Python:** 3.8 o superiore 
- **Java:** Versione 8 (preferibilmente 1.8.0_xxx) JDK
- **PySpark:** 3.2 o superiore
- **Hadoop:** 2.7 o superiore
- **Spark-NLP:** Versione 5.4.1

## Passaggi di Installazione

### 1. Installazione di Java

1. Scarica e installa la versione di Java 8 da [Oracle](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).
2. Imposta la variabile di ambiente `JAVA_HOME`:
   - Vai a `Pannello di Controllo -> Sistema e Sicurezza -> Sistema -> Impostazioni di sistema avanzate -> Variabili d'ambiente`.
   - Crea una nuova variabile di sistema:
     ```
     Nome variabile: JAVA_HOME
     Valore variabile: C:\Program Files (x86)\Java\jre1.8.0_421
     ```
   - Aggiungi `%JAVA_HOME%\bin` alla variabile `Path`.

### 2. Installazione di Hadoop

1. Scarica i file binari di Hadoop per Windows da [WinUtils](https://github.com/steveloughran/winutils) o dalla distribuzione binaria di Hadoop.
2. Crea una directory `C:\hadoop` e copia la cartella `bin` scaricata all'interno:
    C:\hadoop └── bin └── winutils.exe
3. Imposta la variabile di ambiente `HADOOP_HOME`:
    Nome variabile: HADOOP_HOME 
    Valore variabile: C:\hadoop
4. Aggiungi `C:\hadoop\bin` alla variabile `Path`.

### 3. Installazione di Anaconda (Python)

1. Scarica e installa [Anaconda](https://www.anaconda.com/products/distribution) versione 11 o superiore.
2. Crea un nuovo ambiente Python:
```bash
conda create -n pyspark_env python=3.8
conda activate pyspark_env
pip install -r requirements.txt
```

### 4. Risoluzione degli Errori Comuni
* Errore: java.io.FileNotFoundException: Hadoop bin directory does not exist
    - Verifica che la directory C:\hadoop\bin esista e contenga il file winutils.exe.
* Errore: RuntimeError: Java gateway process exited before sending its port number
    - Assicurati che la variabile JAVA_HOME punti alla versione corretta di Java (preferibilmente Java 8).

### Note Aggiuntive
Assicurati che le versioni di PySpark, Spark-NLP e Java siano compatibili tra loro.
Per ulteriori dettagli sulla risoluzione dei problemi, consulta la Documentazione ufficiale di Spark: \href{https://spark.apache.org/docs/latest/}{https://spark.apache.org/docs/latest/} e Spark-NLP: \href{https://nlp.johnsnowlabs.com/}{https://nlp.johnsnowlabs.com/}

### Risorse Utili
WinUtils per Hadoop su Windows: \href{https://github.com/steveloughran/winutils}{https://github.com/steveloughran/winutils}
Guida di Installazione di Spark-NLP: \href{https://nlp.johnsnowlabs.com/docs/en/install}{https://nlp.johnsnowlabs.com/docs/en/install}
