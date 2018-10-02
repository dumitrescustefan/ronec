# -*- coding: utf-8 -*-

import os, copy, sys, json, math
from .core import *
from .conllup import *

# lists all files in a folder, filter by substring (NOT WILDCARD)
def list_files (folder, filename_substring = None):	
	matches = []
	for root, dirnames, filenames in os.walk(folder):
		for filename in filenames:
			if filename_substring != None:
				if filename_substring in filename:
					matches.append(os.path.join(root, filename))
			else:
				matches.append(os.path.join(root, filename))			
	return matches


def read_brat_folder_into_core_format (folder):
    """
    Given a folder will read the .ann file an return a list of Sentence objects.
    It will search in all sub-folders for .ann files
    """
    sentences = []    
    print("Recursively reading BRAT-format files from root folder: [{}]".format(folder))
    ann_files = list_files(folder, filename_substring = ".ann")       
    # read each folder
    for ann_file in ann_files:
        sentences += read_brat_file_into_core_format(ann_file)             
    return sentences
  
def read_brat_file_into_core_format (file): # filename with no extension, will add .txt and .ann automatically
    if file.endswith(".ann"):
        filename = file[:-4]
    else:
        filename = file
    # returns a list of Sentences
    print("\t Reading {} ...".format(filename))
    sentences = []
    sentences_index_start = []
    raw_sentences = ""
    color_sentences = []    
    
    with open(filename+".txt","r") as f:
        raw_sentences = f.read()
    color = 0
    current_sentence = []
    sentence_index_start = 0
    for index, c in enumerate(raw_sentences):        
        if c == "\n":
            color_sentences.append(color)
            color += 1
            sentences.append(Sentence("".join(current_sentence), []))            
            sentences[-1].annotations = []
            current_sentence = []
            sentences_index_start.append(sentence_index_start)
            sentence_index_start = index+1
        else:
            color_sentences.append(color)
            current_sentence.append(c)
        
    with open(filename+".ann","r") as f:        
        ann_data = f.readlines()
    
    for line in ann_data:
        if line[0]=="#":                    
            continue    
        #print(line)
        parts = line.split("\t")
        ann_parts = parts[1].split(" ")
        type = ann_parts[0]
        start = int(ann_parts[1])
        stop = int(ann_parts[2])
        # transpose : 
        transpose_start = start - sentences_index_start[color_sentences[start]]
        transpose_stop = stop - sentences_index_start[color_sentences[start]]
        #print("Start stop {} - {}".format(transpose_start, transpose_stop))
        sentence_id = color_sentences[start]
        sentences[sentence_id].annotations.append(Annotation(transpose_start, transpose_stop, type))
        #sentences[sentence_id].pr()
    
    print("\t\t ... read {} sentences.".format(len(sentences)))
    return sentences
        
def write_brat_format_into_brat_folder(sentences, destination_folder, split, brat_template_conf_files):
    from shutil import copyfile
    
    def _len_list(lst):
        c = 0
        for l in lst:
            c+=len(l)
        return c
    
    print("Writing {} sentences into folder [{}] with {} splits...".format(len(sentences),destination_folder, split))    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        
    if not os.path.exists(os.path.join(brat_template_conf_files,"visual.conf")):
        print("File visual not found in template folder!")
        sys.exit(0)
    if not os.path.exists(os.path.join(brat_template_conf_files,"annotation.conf")):
        print("File annotation not found in template folder!")
        sys.exit(0)
    if not os.path.exists(os.path.join(brat_template_conf_files,"kb_shortcuts.conf")):
        print("File kb_shortcuts not found in template folder!")
        sys.exit(0)
    if not os.path.exists(os.path.join(brat_template_conf_files,"tools.conf")):
        print("File toolsnot found in template folder!")
        sys.exit(0)   
      
    limit_per_split = math.ceil(len(sentences)/split)    
    current_annotation_index = -1
    for split_index in range(1,split+1):
        print("Writing split {}, sentence_index {}/{}".format(split_index,current_annotation_index+1, len(sentences)))
        folder = ""+str(split_index).zfill(2)
        
        # copy stuff        
        if not os.path.exists(os.path.join(destination_folder,folder)):
            os.makedirs(os.path.join(destination_folder,folder))
        copyfile(os.path.join(brat_template_conf_files,"visual.conf"), os.path.join(destination_folder,folder,"visual.conf"))
        copyfile(os.path.join(brat_template_conf_files,"annotation.conf"), os.path.join(destination_folder,folder,"annotation.conf"))
        copyfile(os.path.join(brat_template_conf_files,"kb_shortcuts.conf"), os.path.join(destination_folder,folder,"kb_shortcuts.conf"))
        copyfile(os.path.join(brat_template_conf_files,"tools.conf"), os.path.join(destination_folder,folder,"tools.conf")) 
        
        output_ann = []
        output_txt = []
        char_index = 0
        ann_file_index = 0
        if split_index == split: # put all remaining sentences into last split
            limit_per_split = len(sentences)-current_annotation_index-1 
        
        for _ in range(limit_per_split):
            current_annotation_index+=1 
            # new annotation to write, first the sentence
            char_index = _len_list(output_txt)
            output_txt.append(sentences[current_annotation_index].sentence+"\n")
            for ann in sentences[current_annotation_index].annotations:
                ann_file_index += 1
                type = ann.type
                start = char_index + ann.start
                stop = char_index + ann.stop
                text = sentences[current_annotation_index].sentence[ann.start:ann.stop]
                output_ann.append("T{}\t{} {} {}\t{}\n".format(ann_file_index, type, start, stop, text))
                
        # write files
        with open(os.path.join(destination_folder,folder,"data.txt"), "w") as f:
            for l in output_txt:
                f.write(l)
        with open(os.path.join(destination_folder,folder,"data.ann"), "w") as f:
            for l in output_ann:
                f.write(l)
    print(" Written {} sentences in {} splits".format(current_annotation_index+1, split))
    
def write_core_format_into_conllup_file(sentences, filepath):
    print("Converting {} sentences into CONLLUP format. This requires a text preprocessor for Romanian. If the following function fails please install NLP-Cube (pip3 install nlpcube).")
    
    from cube.api import Cube
    cube=Cube(verbose=True)
    cube.load("ro", tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
    cube_no_tok = Cube(verbose=True)
    cube_no_tok.load("ro", tokenization=False, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
    
    conllupdataset = []
    for sentence in sentences:   
        sentence = process_split_exceptions(sentence)        
        conllupsentence = _conllup_to_core_sentence(sentence, cube, cube_no_tok)        
        conllupdataset.append(conllupsentence)
    
    write_file(filepath, conllupdataset)
    
def read_conllup_file(filepath):
    return read_file(filepath) # from conllup.py
    

def write_conllup_file(conllup_dataset, filepath):
    write_file(filepath, conllup_dataset)
    
def read_conllup_file_into_core_format(filepath):
    dataset = read_conllup_file(filepath)
    sentences = []
    for csentence in dataset:
        text = "{}".format(csentence)
        #print("\n"+text)
        
        # create offsets
        start = 0
        for i in range(len(csentence.tokens)):
            csentence.tokens[i].start = start
            csentence.tokens[i].stop = start + len(csentence.tokens[i].word)                        
            #print("[{}] {}-{}".format(text[csentence.tokens[i].start:csentence.tokens[i].stop], csentence.tokens[i].start, csentence.tokens[i].stop))
            start = csentence.tokens[i].stop + (1 if not "SpaceAfter=No" in csentence.tokens[i].misc else 0)
            
        annotations = []
        i = 0        
        while i < len(csentence.tokens):  
            #print("i={}, offset={}, ent = {}".format(i,csentence.tokens[i].start,csentence.tokens[i].parseme_mwe))
            if csentence.tokens[i].parseme_mwe=="*":                                
                i+=1
            else: # this allows non imbricated entities only.
                #print(csentence.tokens[i].parseme_mwe)
                type = csentence.tokens[i].parseme_mwe[csentence.tokens[i].parseme_mwe.find(":")+1:]
                entity_id = csentence.tokens[i].parseme_mwe[:csentence.tokens[i].parseme_mwe.find(":")]                
                start = csentence.tokens[i].start
                last_token_id = i
                for j in range(i, len(csentence.tokens)):
                    if entity_id not in csentence.tokens[j].parseme_mwe:
                        break
                    else: 
                        last_token_id = j
                stop = csentence.tokens[last_token_id].stop                   
                i = last_token_id + 1
                annotations.append(Annotation(start, stop, type))
                #print(" ANNOTATION: {} text=[{}]".format(annotations[-1], text[start:stop]))
        
        sentences.append(Sentence(text, annotations))
        #input("step")
    return sentences


    
def _conllup_to_core_sentence (sentence_object, cube_object, cube_object_no_tok):
    import sys, copy
    
    print(sentence_object)
    
    sequences=cube_object(sentence_object.sentence)
    annotations = sorted(copy.deepcopy(sentence_object.annotations))
    char_token_id = [-1]*len(sentence_object.sentence)
    
    if len(sequences)>1:
        print(sentence_object.sentence)
        print("ERROR, found more than 1 sentence, but will concatenate all tokens into a single sentence.")
        
        new_sequence = []
        for sequence in sequences:
            for elem in sequence:
                new_sequence.append(elem)
                #print(":: "+elem.word)
        sequences=cube_object_no_tok([new_sequence])
        text = ""
        for elem in sequences[0]:
            text+=elem.word
            if not "SpaceAfter=No" in elem.space_after:
                text+=" "
        #print(">"+text+"<")
    
    sequence = sequences[0]
    index = 0
    
    # mark each char position with its token id
    for token_id, token in enumerate(sequence):
        token.parseme_mwe = ""
        for i in range(len(token.word)):
            char_token_id[index+i] = token_id
        index+=len(token.word)
        if not "SpaceAfter=No" in token.space_after:
            char_token_id[index] = token_id
            index+=1
    #print(" ".join([str(x) for x in char_token_id]))
    
    # iterate through all annotations and mark for each what tokens it encompasses   
    for i in range(len(annotations)):
        annotations[i].token_ids = set()
        for j in range(annotations[i].start, annotations[i].stop):
            annotations[i].token_ids.add(char_token_id[j])
        annotations[i].token_ids = sorted(list(annotations[i].token_ids))
        #print("Annotation {}: {}-{}-{} has token_ids {}".format(i, annotations[i].start, annotations[i].stop, annotations[i].type, annotations[i].token_ids))
        #txt = "\t"
        #for token_id in annotations[i].token_ids:
        #    txt += " "+ sequence[token_id].word
        #print(txt)
    
    # write for each annotation the appropriate string in the parseme_mwe filed        
    for i in range(len(annotations)):
        for id in annotations[i].token_ids:
            if id == annotations[i].token_ids[0]: # first entry
                sequence[id].parseme_mwe+=";"+str(i+1)+":"+annotations[i].type
            else:
                sequence[id].parseme_mwe+=";"+str(i+1)
                
    # each empty entry should have a "_", cleanup ";"
    for token in sequence:
        if len(token.parseme_mwe)==0:
            token.parseme_mwe = "*"
        elif token.parseme_mwe[0] == ";":
            token.parseme_mwe = token.parseme_mwe[1:]

    # fill a CONLLUPSentence object    
    conllupsentence = CONLLUPSentence()
    for token_id, t in enumerate(sequence):        
        conlluptoken = Token(index=token_id+1, word=t.word, lemma=t.lemma, upos=t.upos, xpos=t.xpos, feats=t.attrs, head=t.head, deprel=t.label, deps=t.deps, misc=t.space_after, parseme_mwe=t.parseme_mwe)
        conllupsentence.tokens.append(conlluptoken)
    
    print(conllupsentence)
    for line in conllupsentence.to_text():
        print(line)
    
    return conllupsentence
   
       
    
    
    