# Foresta_a_caso

Per il mio progetto di Intelligenza Artificiale ho implementato l’algoritmo top-down per l’apprendimento di alberi di decisione descritto in classe ed in R&N 2010 e l’ho modificato per implementare Random Forest come esposto brevemente in classe ed in maggiore dettaglio in (Hastie et al. 2017, 15.2). L'algoritmo è implementato solo per lo scopo della classificazione. Ho verificato il funzionamento dell’algoritmo sviluppato comparando i risultati con quelli ottenuti tramite l’implementazione disponibile in scikit-learn, su almeno tre data sets scelti a piacere dall’UCI Machine Learning Repository. Infine ho riportato i risultati dei confronti delle due implementazioni.

Un maggiore approfondimento e i risultati sono riportati nel file Risultati.pdf

## Requirements

Per utilizzare questo progetto, è necessario installare le seguenti librerie (tuttavia non tutti gli script utilizzano necessariamente tutte le librerie, quindi in base alle dipendenze dello script si può decidere di non installare alcune librerie):

### Pandas

''' pip install pandas ''' installazione tramite pip

### Scikit-learn

''' pip install -U scikit-learn ''' installazione tramite pip

### Seaborn

''' pip install seaborn ''' installazione tramite pip

### PreattyTable

''' python3 -m pip install -U prettytable '''   ----->     https://pypi.org/project/prettytable/

### tqdm

''' pip install tqdm ''' installazione tramite pip

### Graphviz

In alcuni casi la libreria da un po' di problemi, consiglio di installare dal pacchetto apt (in caso si usi windows sono necessari ulteriori passaggi vedi la pagina di graphviz):

''' apt install graphviz '''  Installare il pacchetto

### matplotlib

''' python -m pip install -U pip '''
''' python -m pip install -U matplotlib '''

## treelearning.py e datanalisys.py
Allinterno del file treelearning.py si può trovare le mie implementazioni di DecisionTreeClassifier e RandomForestClassifier.
Nel file datanalisys.py sono presenti i metodi che permettono di visualizzare i dati dei database e modificarli se necessario. (vedi script datavisualizzation.py)

## Utilizzo

Per utilizzare questo progetto, è necessario clonare il repository e installare le dipendenze necessarie:

''' git clone https://github.com/GianniMoretti/Foresta_a_caso.git '''

### Risultati Decision Tree
Per replicare i risultati (immagini degli alberi) presenti nel file Relazione.pdf si deve eseguire lo script nominato my_d_tree.py:

''' python my_d_tree.py '''  Per eseguire il comando si deve essere nella caartella dove è contenuto il file.

Questo script mostra l'albero di decisione su graphviz, le matrici di confusione di train e test e il classification report su train e test.
Per poter cambiare il database su cui fare la prova, è necessario cambiare i dati relativi al database che sono scritti all'inizio dello script.

Il file skl_d_tree.py fa la stessa cosa ma utilizzando l'implementazione DecisionTreeClassifier di sklearn.

### Risultati My RandomForest vs Sklearn RandomForest

Per replicare i risultati (grafici) presenti nel file Relazione.pdf si deve eseguire lo script nominato my_vs_skl_r_forest.py:

''' python my_vs_skl_r_forest.py '''  Per eseguire il comando si deve essere nella caartella dove è contenuto il file.

Per poter cambiare il database su cui fare la prova, è necessario cambiare i dati relativi al database che sono scritti all'inizio dello script.
Per semplicità, i dati relativi ai database da me scelti possono essere trovati all'interno del file database.txt, così da poterli copiare e sostituire direttamente nel file python my_vs_skl_r_forest.py.

## Licenza

Questo progetto è concesso in licenza sotto la licenza MIT - vedere il file [LICENSE.md](LICENSE.md) per i dettagli.