# Autonomic Semantic Interpreter

### Descrizione del Progetto
Questo progetto utilizza PySpark e la libreria Spark NLP per implementare un modello <b/>RAG (Retrieval-Augmented Generation)</b> orientato all'analisi semantica di frasi in italiano. L'obiettivo è identificare argomenti comuni tra le frasi utilizzando tecniche di elaborazione del linguaggio naturale e machine learning. I principali passaggi del flusso di lavoro sono i seguenti:

1. **Sentence Embedding con BERT:** Viene utilizzato un modello BERT pre-addestrato per generare embedding delle frasi in italiano, trasformandole in vettori numerici che catturano il loro significato semantico.

2. **Riduzione della Dimensione con PCA:** Per ottimizzare il clustering e ridurre la complessità computazionale, i vettori delle frasi vengono ridotti dimensionalmente utilizzando l'analisi delle componenti principali (PCA).

3. **Clustering:** I vettori ridotti vengono raggruppati in cluster utilizzando algoritmi di clustering non supervisionati, al fine di identificare frasi simili che condividono concetti semantici.

4. **Analisi del Contenuto con TF-IDF:** Una volta formati i cluster, si utilizza la tecnica TF-IDF per analizzare il contenuto delle frasi in ciascun cluster, estraendo il valore semantico dominante che rappresenta l'argomento comune discusso.

Questo approccio consente di comprendere e organizzare automaticamente le frasi in base ai temi trattati, facilitando l'analisi del linguaggio naturale e l'estrazione di informazioni utili da insiemi di testi non strutturati.
