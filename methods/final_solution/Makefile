PYTHON=python

COLUMN_ARGUMENT=-column author
REVERSE_ARGUMENT=-reverse true

SCRIPT_FILTER=../../scripts/filter_submission.py
SCRIPT_REINDEX=../../scripts/reindex_column.py

ROOT_FOLDER=../../custom_data/
FOLDER_REINDEXED=$(ROOT_FOLDER)reindexed/

# ------------------- Global rules ---------------------------------

init:graph_metrics.csv

exec:submission.csv

postprocessing:submission_postprocessed.csv submission_doc2vec_postprocessed.csv submission_normal_postprocessed.csv

all:preprocessing

# ------------------- Cleaning rules ---------------------------------

clean:
	rm -rf *.csv

clean_windows:
	del *.csv

# ------------------- Specific rules ---------------------------------

graph_metrics.csv: $(FOLDER_REINDEXED)coauthorship_undirected.csv $(FOLDER_REINDEXED)authors_abstracts_count.csv
	$(PYTHON) graph_metrics.py $^ $@

output_normal.csv: $(FOLDER_REINDEXED)authors_features.csv graph_metrics.csv $(FOLDER_REINDEXED)coauthorship_undirected.csv $(FOLDER_REINDEXED)train.csv $(FOLDER_REINDEXED)validation.csv
	$(PYTHON) gnn.py $(word 1,$^) $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^) $@

output_doc2vec.csv: $(FOLDER_REINDEXED)authors_features_doc2vec.csv graph_metrics.csv $(FOLDER_REINDEXED)coauthorship_undirected.csv $(FOLDER_REINDEXED)train.csv $(FOLDER_REINDEXED)validation.csv
	$(PYTHON) gnn.py $(word 1,$^) $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^) $@

output.csv:output_normal.csv output_doc2vec.csv $(FOLDER_REINDEXED)train.csv $(FOLDER_REINDEXED)validation.csv $(FOLDER_REINDEXED)test.csv
	$(PYTHON) aggregate.py $(word 1,$^) $(word 2,$^) $(word 3,$^) $(word 4,$^) $(word 5,$^) $@

submission_normal.csv: output_normal.csv
	$(PYTHON) $(SCRIPT_REINDEX) $(COLUMN_ARGUMENT) $(REVERSE_ARGUMENT) $< $(FOLDER_REINDEXED)reindex_dictionnary.csv $@
	$(PYTHON) $(SCRIPT_FILTER) $@ $(ROOT_FOLDER)test.csv $@

submission_doc2vec.csv: output_doc2vec.csv
	$(PYTHON) $(SCRIPT_REINDEX) $(COLUMN_ARGUMENT) $(REVERSE_ARGUMENT) $< $(FOLDER_REINDEXED)reindex_dictionnary.csv $@
	$(PYTHON) $(SCRIPT_FILTER) $@ $(ROOT_FOLDER)test.csv $@

submission.csv: output.csv
	$(PYTHON) $(SCRIPT_REINDEX) $(COLUMN_ARGUMENT) $(REVERSE_ARGUMENT) $< $(FOLDER_REINDEXED)reindex_dictionnary.csv $@
	$(PYTHON) $(SCRIPT_FILTER) $@ $(ROOT_FOLDER)test.csv $@

submission_normal_postprocessed.csv: submission_normal.csv
	$(PYTHON) postprocess.py $< $(FOLDER_REINDEXED)authors_abstracts_count.csv $(ROOT_FOLDER)test.csv $@

submission_doc2vec_postprocessed.csv: submission_doc2vec.csv
	$(PYTHON) postprocess.py $< $(FOLDER_REINDEXED)authors_abstracts_count.csv $(ROOT_FOLDER)test.csv $@

submission_postprocessed.csv: submission.csv
	$(PYTHON) postprocess.py $< $(FOLDER_REINDEXED)authors_abstracts_count.csv $(ROOT_FOLDER)test.csv $@

.ALWAYSDO :