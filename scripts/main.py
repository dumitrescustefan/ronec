"""
Example file
"""

import sys, os, copy
from convertors.core import *
import convertors.brat as brat
import convertors.conllup as conllup


print("Example 1: BRAT format")
print("_"*40+"\n")


sentences = brat.read_folder(os.path.join("data","brat"))
print("\nRead {} sentences.".format(len(sentences)))
print("First sentence is: \n\n{}".format(sentences[0]))    

# correct sentence
from convertors.core import process_split_exceptions
new_sentence = process_split_exceptions(sentences[0])
print("Corrected sentence: \n\n{}".format(new_sentence))    

#print("Writing into CONLLUP format ...")
#conllup.write_file("temporary.conllup", sentences)

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
                    errors += 1
                    conllupdataset.append(conllupsentence)
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