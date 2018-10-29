
# A list of all the folders that have makefiles in them.  Running make all builds all these things
SUBDIRS = tools/ner_stream examples/C/ner examples/C/relation_extraction examples/cpp/ner examples/cpp/train_ner \
	  examples/cpp/train_relation_extraction examples/cpp/relation_extraction

examples: tools/ner_stream examples/C/ner examples/C/relation_extraction
	cp examples/C/ner/ner_example .
	cp examples/C/relation_extraction/relation_extraction_example .
	cp tools/ner_stream/ner_stream .

MITIE-models:
	wget -O - http://sourceforge.net/projects/mitie/files/binaries/MITIE-models-v0.2.tar.bz2 > MITIE-models-v0.2.tar.bz2
	tar -xjf MITIE-models-v0.2.tar.bz2

test: all examples MITIE-models
	./ner_stream MITIE-models/english/ner_model.dat < sample_text.txt > /tmp/MITIE_test.out
	diff /tmp/MITIE_test.out sample_text.reference-output
	./relation_extraction_example MITIE-models/english/ner_model.dat MITIE-models/english/binary_relations/rel_classifier_location.location.contains.svm sample_text.txt > /tmp/MITIE_test_rel.out
	diff /tmp/MITIE_test_rel.out sample_text.reference-output-relations
	@echo Testing completed successfully



.PHONY: mitie mitielib $(SUBDIRS)
all: $(SUBDIRS)
mitie: mitielib
$(SUBDIRS): mitie 
	$(MAKE) -C $@
mitielib: dlib/dlib
	$(MAKE) -C $@
dlib/dlib:
	git submodule update --init
clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
	@rm -rf ner_stream ner_example relation_extraction_example
