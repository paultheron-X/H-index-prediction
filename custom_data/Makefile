PYTHON=python
SCRIPT_FOLDER=../scripts/
SCRIPT_FILTER_EDGELIST=../scripts/init_coautorship_edgelist.py
SCRIPT_FORMAT_AUTHORS_ABSTRACTS=../scripts/
SCRIPT_TRAIN_W2V=../scripts/w2v_train.py
SCRIPT_APPLY_W2V=../scripts/w2v_apply.py
SCRIPT_APPLY_W2V_PRETRAINED=../scripts/w2v_apply_pretrained.py
SCRIPT_REINDEX=../scripts/reindex_column.py
SCRIPT_COMPUTE_STATS=../scripts/compute_stats.py

VECTOR_SIZE_OPTION=-v 300
TEST_PROPORTION_OPTION=-t 0.05

all:format reindex

clean_windows:
	del *.csv
	del *.wordvector

clean:
	rm -rf *.csv
	rm -rf *.wordvector

clean_rec:clean
	cd reindexed && $(MAKE) clean

# ------------- Formating rules ---------------------------

format:authors_abstracts.csv coauthorship_undirected.csv abstracts.csv abstract_features.csv format_train_validation_test authors_features.csv

format_authors_abstracts: authors_abstracts.csv

format_coauthorship: coauthorship_undirected.csv

format_abstracts: abstracts.csv

generate_abstracts_vectors : abstract_features.csv

join_authors_abstracts: authors_features.csv

w2v_train: model.wordvector

w2v_apply : abstract_features.csv

w2v_apply_pretrained : abstract_features_pretrained.csv

abstracts.csv: ../raw_data/abstracts.txt
	 $(PYTHON) $(SCRIPT_FOLDER)format_abstracts.py $< $@

format_train_validation_test : ../raw_data/train.csv ../raw_data/test.csv 
	 $(PYTHON) $(SCRIPT_FOLDER)format_train_validation_test.py $(TEST_PROPORTION_OPTION) $^ train.csv validation.csv test.csv

authors_abstracts.csv : ../raw_data/author_papers.txt
	 $(PYTHON) $(SCRIPT_FOLDER)format_authors_abstracts.py $^ $@ authors_abstracts_count.csv

coauthorship_undirected.csv : coauthorship.csv
	$(PYTHON) $(SCRIPT_FOLDER)generate_edges_undirected.py $< $@

coauthorship.csv : ../raw_data/coauthorship.edgelist
	 $(PYTHON) $(SCRIPT_FOLDER)format_coauthorship.py $^ $@

model.wordvector: abstracts.csv
	 $(PYTHON) $(SCRIPT_FOLDER)w2v_train.py $(VECTOR_SIZE_OPTION) $< $@

abstract_features.csv : abstracts.csv model.wordvector
	 $(PYTHON) $(SCRIPT_FOLDER)w2v_apply.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $@ $(word 2,$^)

authors_features.csv: authors_abstracts.csv abstract_features.csv 
	 $(PYTHON) $(SCRIPT_FOLDER)join_authors_abstracts.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $(word 2,$^) $@

authors_features_doc2vec.csv: authors_abstracts.csv doc2vec_features.csv 
	 $(PYTHON) $(SCRIPT_FOLDER)join_authors_abstracts.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $(word 2,$^) $@

doc2vec_features.csv: abstracts.csv 
	$(PYTHON) $(SCRIPT_FOLDER)generate_doc2vec_embedding.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $@

roberta_features.csv: abstracts.csv
	$(PYTHON) $(SCRIPT_FOLDER)generate_roberta_embedding.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $@ 

node2vec_features.csv: edge_index/coauthorship_undirected.csv 
	$(PYTHON) $(SCRIPT_FOLDER)generate_node2vec_embedding.py $(VECTOR_SIZE_OPTION) $(word 1,$^) $@ n2v.model


# -----------Reindexing rules-----------------------------------

reindex:
	 cd reindexed && $(MAKE) all

# ------------Usefull-------------------------------------------

compute_stats:../raw_data/train.csv
	 $(PYTHON) $(SCRIPT_FOLDER)compute_stats.py $< hindex

w2v_test: model.wordvector
	 $(PYTHON) $(SCRIPT_FOLDER)w2v_test.py $(VECTOR_SIZE_OPTION) $<