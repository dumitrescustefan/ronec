# -*- coding: utf-8 -*-

import sys, os, copy
from convertors import scripts
from convertors.core import *
import convertors.conllup as conllup


DATA_FOLDER = os.path.abspath(os.path.join("..","ronec"))
BRAT_FOLDER = os.path.join(DATA_FOLDER,"brat","ronec")
BRAT_TEMPLATES = os.path.join(DATA_FOLDER,"brat","template")
CONLLUP_FILE = os.path.join(DATA_FOLDER,"conllup","ronec.conllup")
SCRATCH_FOLDER = os.path.abspath(os.path.join("..","temp"))
SCRATCH_FOLDER_CONLLUP_FILE = os.path.abspath(os.path.join("..","temp","example.conllup"))


# ############################################################################
print("\n\t 1. Read from BRAT folder into core format\n"+"_"*60)
core_sentences = scripts.read_brat_folder_into_core_format(BRAT_FOLDER)
print("Done, read {} sentences.".format(len(core_sentences)))


# ############################################################################
print("\n\t 2. Write from core format to BRAT format\n"+"_"*60)
print("\t This function will write all the core.Sentence objects into root folder {}, where it will create {} sub-folders (named as incremental integers), each with and equal number of sentences. It needs the files from {} where the .conf files are located in order to be a valid BRAT folder. To import it in BRAT, simply copy the folder to the /data/ folder in your BRAT installation.".format(SCRATCH_FOLDER, 20, BRAT_TEMPLATES))
scripts.write_brat_format_into_brat_folder(core_sentences, SCRATCH_FOLDER, split = 20, brat_template_conf_files = BRAT_TEMPLATES)


# ############################################################################
print("\n\t 3. Write from core format into CONLLUP file\n"+"_"*60)
print("Warning, this can break if a BRAT sentence has multiple sentences in it (as they will be detected by NLP-Cube accordingly. The script below will concatenate these sentences into a single one. This will produce a valid CONLLUP file, even though it contradicts the one-sentence at a time 'contract' of the CONLLUP format. Please ensure that BRAT only contains single sentences, as the CONLLUP conversion process will be difficult later on. Here, we only convert 2 sentences for demo purposes.")
scripts.write_core_format_into_conllup_file(core_sentences[0:2], SCRATCH_FOLDER_CONLLUP_FILE)


# ############################################################################
print("\n\t 4. Read from CONLLUP file into CONLLUP format\n"+"_"*60)
conllup_dataset = scripts.read_conllup_file(CONLLUP_FILE)
print("Done, read {} sentences.".format(len(conllup_dataset)))


# ############################################################################
print("\n\t 5. Write from CONLLUP format into CONLLUP file\n"+"_"*60)
scripts.write_conllup_file(conllup_dataset,SCRATCH_FOLDER_CONLLUP_FILE)
print("Done, written {} sentences.".format(len(conllup_dataset)))


# ############################################################################
print("\n\t 6. Read from CONLLUP file into core format\n"+"_"*60)
core_sentences = scripts.read_conllup_file_into_core_format(CONLLUP_FILE)
print("Done, read {} sentences.".format(len(core_sentences)))

# ############################################################################
print("\n\t 7. Example of the core format:\n"+"_"*60)
sentence = core_sentences[0] # select first sentence
print("First sentence is:")
print(sentence)
print("First sentence text: {}".format(sentence.sentence))
print("First sentence annotations (list of core.Annotation objects): {}".format(sentence.annotations))
print("Example of an Annotation: type={}, start_char={}, stop_char={}, text_from_sentence={}".format(sentence.annotations[0].type,sentence.annotations[0].start, sentence.annotations[0].stop, sentence.sentence[sentence.annotations[0].start:sentence.annotations[0].stop] ) )


print("Here are some overall statistics of the corpus:")
count = 0
per_entity_dict = {}
for sentence in core_sentences:
    count += len(sentence.annotations)
    for ann in sentence.annotations:
        if ann.type not in per_entity_dict:
            per_entity_dict[ann.type] = 0
        per_entity_dict[ann.type] = per_entity_dict[ann.type] + 1
print("\t In total, we have annotated {} entities. Here's a breakdown of per-entity counts:".format(count))
for key, val in per_entity_dict.items():
    print("\t\t {} \t: \t{}".format(key,val))


# ############################################################################
print("\n\t 8. Example of the CONLLUP format:\n"+"_"*60)
sentence = conllup_dataset[0]
print("First CONLLUPSentence object is:")
print(sentence)
print("List of first tokens:")
for i in range(min(10,len(sentence.tokens))):
    token = sentence.tokens[i]
    line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
            token.index,
            token.word,
            token.lemma,
            token.upos,
            token.xpos,
            token.feats,
            token.head,
            token.deprel,
            token.deps,
            token.misc,
            token.parseme_mwe)
    print(line)

    
# ############################################################################
print("\n\t 9. Example of sentence tokenization 'correction':\n"+"_"*60)

from convertors.core import process_split_exceptions
print("Issues arise when we import from BRAT (which has no knowledge of sentence segmentation or tokenization) into CONLLUP format .. which has. There is one notable exception that breaks the conversion from BRAT to CONNLU: tokens that belong to different entities that are separated by non-space characters. Example: 24-24 are two numbers separated by a hyphen. They do not tokenize well, meaning that instead of '24', '-', '24' (3 tokens), there is only one token '24-24' with two entities in them, making the conversion impossible. Therfore, apply the 'process_split_exceptions' function with core.Sentence parameter to introduce spaces between these entities and make possible the conversion. If this function fails with 'dtw' not found error, please install 'pip3 install dtw'.")
print("For example let's consider the sentence: ")    
# correct sentence
old_sentence = Sentence(sentence="24-24 este scorul curent.", annotations=[Annotation(0,2,"NUMERIC_VALUE"), Annotation(3,5,"NUMERIC_VALUE")])
print(old_sentence)
print("Processing this sentence yields: (note the spaces introduced around the hyphen) ")
new_sentence = process_split_exceptions(old_sentence)
print(new_sentence)    
