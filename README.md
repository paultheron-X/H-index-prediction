# INF554---H-index-prediction

## Report
Please find at the root of the repository a report describing our data pipeline and neural networks.

## Execution

To test the algorithms, you first have to load the data. Put the following files in the *raw_data* folder :
*   abstracts.txt
*   authors_papers.txt
*   coauthorship.edgelist
*   test.csv
*   train.csv

Then, execute the following commands

 ```bash
cd custom_data
make all
```
If everything completed properly, you should have all the files in your *custom_data* and *custom_data/reindexed* folder. 

To execute our method, type

```bash
cd methods/final_solution
make init
make exec
```

Here, the **init** rule initialise the graph metrics, the **exec** rule runs the GNNs and the **postprocessing** rule applies the postprocessing to the output of the previous rule. The result is *submission.csv*.

Estimated times :   
*   Formating data : 2h
*   Running the method (init + exec + postprocess) : 3min with a Nvidia GPU

## Folders

**Custom data :** 
Contains the data formated and processed. The sub-flder reindexed contains the same data, with reindexed authors (usefull for pytorch_geometric graph model).

**Defaukt script submissions :**
Contains the default methods that compute basic solutions. Based on kaggle files (unedited).

**Methods :**
Contains one folder per method use to solve the problem. The "submission.csv" files are inside each folder. Trash folder contains methods we ended up not using.

**Scripts :**
Contains the python scripts used to format and process data.
