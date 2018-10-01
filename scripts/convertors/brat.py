# -*- coding: utf-8 -*-

# import from brat
# export to brat
# merge folders
# split into folders

import os    
import copy
from .core import *
from .util import list_files


def read_folder (folder):
    """
    Given a folder will read the .ann file an return a list of Sentence objects.
    It will search in all sub-folders for .ann files
    """
    sentences = []    
    print("Recursively reading BRAT-format files from root folder: [{}]".format(folder))
    ann_files = list_files(folder, filename_substring = ".ann")       
    # read each folder
    for ann_file in ann_files:
        sentences += read_file(ann_file)             
    return sentences
  
def read_file (file): # filename with no extension, will add .txt and .ann automatically
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
    