PYTHON=python
SCRIPT_FOLDER = ../../scripts/

all: reindex_dictionnary.csv authors_abstracts.csv authors_abstracts_count.csv coauthorship_undirected.csv coauthorship.csv train.csv validation.csv test.csv authors_features.csv

clean:
	rm -rf *.csv

generate_reindex_dictionnary: reindex_dictionnary.csv

reindex_dictionnary.csv : ../authors_abstracts.csv
	$(PYTHON) $(SCRIPT_FOLDER)generate_reindex_dictionnary.py $< $@

authors_abstracts.csv : ../authors_abstracts.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^)  $(word 2,$^) $@ 

authors_abstracts_count.csv : ../authors_abstracts_count.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^)  $(word 2,$^) $@ 

# authors_features.csv : ../authors_features.csv reindex_dictionnary.csv
authors_features.csv : ../authors_features.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^)  $(word 2,$^) $@ 

authors_features_doc2vec.csv : ../authors_features_doc2vec.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^)  $(word 2,$^) $@ 

coauthorship_undirected.csv: ../coauthorship_undirected.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author_1 $(word 1,$^)  $(word 2,$^) $@ 
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author_2 $@  $(word 2,$^) $@ 

coauthorship.csv : ../coauthorship.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author_1 $(word 1,$^)  $(word 2,$^) $@ 
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author_2 $@  $(word 2,$^) $@ 

train.csv : ../train.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^) $(word 2,$^) $@

validation.csv : ../validation.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^) $(word 2,$^) $@

test.csv : ../test.csv reindex_dictionnary.csv
	$(PYTHON) $(SCRIPT_FOLDER)reindex_column.py -column author $(word 1,$^) $(word 2,$^) $@