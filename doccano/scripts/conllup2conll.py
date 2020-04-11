#!/usr/bin/env python
#encoding=utf-8
'''
Converter from .conllu format to .conll format for Universal Dependencies
Check if each folder has three .conllu files for train, dev and test, and if the files are properly filled (e.g.: word or lemma are _)
Command line:
python conllu-to-conll-standard.py #<origin path> #<destination path>
'''

import sys

with open(sys.argv[1].strip(),'r') as conllu, open(sys.argv[2].strip(),'w') as conll:

    if len(sys.argv) < 3:
        print("Please select input and output files")
        sys.exit()
    if len(sys.argv) == 3:
        selected_column = 10
    else:
         selected_column = int(sys.argv[3])
         if selected_column > 10:
             print("Please select column number < 11")
             sys.exit()

    lines=conllu.readlines()

    for line in lines:
        tupl = line.split()
	#extract the word and last tag from conllu file
        if len(tupl)==11 and tupl[0]!='#' and '.' not in tupl[0] and '-' not in tupl[0]:
            print("Length of tuple is {}".format(len(tupl)))
            conll.write(tupl[1]+'\t'+tupl[selected_column]+'\n')
        if len(tupl)==0:
            conll.write('\n')
