# -*- coding: utf-8 -*-

# CoNLL-U Plus
import os    
import copy
from .core import *
from .util import list_files

class Token(object):
    def __init__ (self, index=-1, word="_", lemma="_", upos="_", xpos="_", feats="_", head="_", deprel="_", deps="_", misc="_", parseme_mwe="_"):
        self.index, self.is_compound_entry = self._int_try_parse(index)
        self.word = word
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head, _ = self._int_try_parse(head)
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        self.parseme_mwe = parseme_mwe        
        
    def _int_try_parse(self, value):
        try:
            return int(value), False
        except ValueError:
            return value, True  

class CONLLUPSentence(object):
    def __init__ (self, id = None):
        self.tokens = []
        self.id = id
    
    def __repr__(self):
        sentence = ""
        for token in self.tokens:
            sentence += token.word
            if not "SpaceAfter=No" in token.misc:
                sentence += " "
        return sentence

    def to_text(self):
        lines = []
        if self.sentence_id != None:
            lines.append("# sent_id = {}\n".format(self.sent_id))
        lines.append("# text = {}\n".format(self))
        for token in self.tokens:
            lines.append("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
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
            token.parseme_mwe            
            ))
        return lines

"""
This processes sentences from the core format to the conllup format, including annotation
"""        
def process_sentence (sentence_object, cube_object, force_single = False, cube_object_no_tok = None):
    import sys, copy
    
    # change „ and “ , needed do to tokenizer exceptions in current model
    sentence_object.sentence = sentence_object.sentence.replace("“","\"").replace("„","\"")
    
    print(sentence_object)
    
    sequences=cube_object(sentence_object.sentence)
    annotations = sorted(copy.deepcopy(sentence_object.annotations))
    char_token_id = [-1]*len(sentence_object.sentence)
    
    if not force_single:
        if len(sequences)>1:
            print(sentence_object.sentence)
            print("ERROR, found more than 1 sentence here!")
            return None        
    else:                
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
        print(">"+text+"<")
    
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
    print(" ".join([str(x) for x in char_token_id]))
    
    # iterate through all annotations and mark for each what tokens it encompasses   
    for i in range(len(annotations)):
        annotations[i].token_ids = set()
        for j in range(annotations[i].start, annotations[i].stop):
            annotations[i].token_ids.add(char_token_id[j])
        annotations[i].token_ids = sorted(list(annotations[i].token_ids))
        print("Annotation {}: {}-{}-{} has token_ids {}".format(i, annotations[i].start, annotations[i].stop, annotations[i].type, annotations[i].token_ids))
        txt = "\t"
        for token_id in annotations[i].token_ids:
            txt += " "+ sequence[token_id].word
        print(txt)
    
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
   
"""
This function reads a conllup file and returns the results as an array of CONLLUPSentences
"""
def read_file (file):
    with open(filename+".txt","r") as f:
        raw_sentences = f.read()
    return []
  
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
    
    
    color = 0
    current_sentence = []
    sentence_index_start = 0
    for index, c in enumerate(raw_sentences):        
        if c == "\n":
            color_sentences.append(color)
            color += 1
         
    
# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC PARSEME:MWE    
def write_file (filename, sentences):
    output = open(filename,"w",encoding="utf8")
    
    for sentence_id, sentence in enumerate(sentences):
        
        pass
    
    output.close()
    
