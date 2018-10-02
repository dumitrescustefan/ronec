# -*- coding: utf-8 -*-

"""
Note, this was used during internal text alignments, we store it for future reference but no function from here is further needed.
"""

import os    
import copy
import core
from util import list_files

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
   
   
   
   
   
   
   
   
exceptions = ['s', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'm', 'm', 's', 's', 's', 's', 's', 'm', 'm', 's', 's', 's', 'm', 's', 's', 'x', 'm', 's', 's', 'm', 's', 's', 's', 'm', 'm', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 'm', 'm', 'm', 'm', 'm', 's', 'm', 'm', 'm', 'm', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'x', 's', 's', 'm', 's', 'm', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 'm', 's', 's', 'm', 's', 'm', 's', 'm', 's', 'x', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'm', 'm', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 'm', 'm', 'x', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 'm', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 'm', 'x', 's', 's', 'x', 's', 's', 's', 's', 's', 'm', 's', 's', 'm', 's', 'm', 'm', 'm', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'm', 'm', 's', 's', 'm', 's', 'm', 's', 'm', 's', 's', 'm', 'm', 'm', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 'm', 's', 'm', 's', 'm', 's', 'x', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 's', 'm', 's', 's', 's', 'm', 'm', 's', 's', 's', 's', 's', 's', 's', 's', 'm', 's', 'm', 'm', 's', 's', 'm', 's', 's', 's', 'm', 'm', 's', 's', 'm', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 's', 'm', 's', 's', 'm', 's', 's', 's', 's', 's', 's', 'm', 's', 'm', 's', 's', 's', 's', 'm', 's', 's', 's', 'm', 'm', 's', 's', 's', 'm']

from cube.api import Cube
cube=Cube(verbose=True)
cube.load("ro", tokenization=True, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
cube_no_tok = Cube(verbose=True)
cube_no_tok.load("ro", tokenization=False, compound_word_expanding=False, tagging=True, lemmatization=True, parsing=True)
multi_sentence_count = 0
errors = 0
multi_sentences = []
conllupdataset = []
for sentence in sentences:
    if "Alege: [" in sentence.sentence or "Decide tipul/clasa corecta: [" in sentence.sentence or len(sentence.sentence)==0:
        continue
    
    #print("BEF split exc:")
    #print(sentence)
    sentence = process_split_exceptions(sentence)
    #print("AFT split exc:")
    #print(sentence)
    
    conllupsentence = conllup.process_sentence(sentence, cube)
    if conllupsentence == None:
        decision = exceptions[multi_sentence_count]
        multi_sentence_count +=1
        if decision == "s":
            conllupsentence = conllup.process_sentence(sentence, cube, force_single = True, cube_object_no_tok = cube_no_tok)
            if conllupsentence != None:
                errors += 1
                conllupdataset.append(conllupsentence)
            #input("check")
        if decision == "m":
            # need to split annotations in several arrays
            sequence_offset = 0            
            sequences=cube(sentence.sentence)
            copy_annotations = sorted(copy.deepcopy(sentence.annotations))
            for sequence in sequences:
                text = ""
                for elem in sequence:
                    text+=elem.word
                    if not "SpaceAfter=No" in elem.space_after:
                        text+=" "
                annotations = []
                i = 0
                while i<len(copy_annotations):                                        
                    print("\t\t current ann is: {}, len text = {}, offset = {}".format(copy_annotations[i], len(text), sequence_offset))
                    if copy_annotations[i].stop <= sequence_offset+len(text):                        
                        ann = copy.deepcopy(copy_annotations[i])
                        ann.start = ann.start - sequence_offset
                        ann.stop = ann.stop - sequence_offset
                        print("\t\t\t Tranferred ann is : {}".format(ann))
                        annotations.append(ann)
                        copy_annotations.remove(copy_annotations[i])
                        print("\t... copy_ann has len {}, ann has len {}".format(len(copy_annotations), len(annotations)))
                        i=0
                    else:
                        i+=1
                        
                print("Copy_ann has len {}, ann has len {}".format(len(copy_annotations), len(annotations)))
                
                partial_sentence = Sentence(text.strip(), annotations)
                print(partial_sentence)
                partial_sentence = process_split_exceptions(partial_sentence)
                sequence_offset += len(text)
                
                #print("PR: ["+partial_sentence.sentence+"]") 
                conllupsentence = conllup.process_sentence(partial_sentence, cube)
                if conllupsentence != None:                    
                    conllupdataset.append(conllupsentence)
                else:
                    errors += 1
                #input("check")
        """
        multi_sentence_count +=1
        
        sequences = cube(sentence.sentence)
        if ":" in sequences[0][-1].word:
             multi_sentences.append("s")
        else:            
            print("______________________________________")
            print("ORIGNIAL: ")
            print("\n\033[44m"+sentence.sentence+"\033[0m")            
            for sequence in sequences:
                text = ""
                for elem in sequence:
                    text+=elem.word+" "
                print("\t"+text.strip())
            print()           
            
            decision = input("s - treat as single, m - treat as multi, x - discard: ")        
            multi_sentences.append(decision)
        #if len(multi_sentences)>2:
        #    break
        """
    else:
        conllupdataset.append(conllupsentence)   

print(multi_sentences)        
print("Total missed sentences : "+str(multi_sentence_count))        
print("Total errors : "+str(errors))        
#for sent in multi_sentences:
#    print(sent+"\n")
    
conllup_file_handle = open("dataset.conllup","w")        
for sentence_id, sentence in enumerate(conllupdataset):
    sentence.id = sentence_id + 1
    for line in sentence.to_text():    
        conllup_file_handle.write(line)
    conllup_file_handle.write("\n")
conllup_file_handle.close()