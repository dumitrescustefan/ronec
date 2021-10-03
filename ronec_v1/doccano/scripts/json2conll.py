#!/usr/bin/env python
#encoding=utf-8

import sys
import codecs
import json

if len(sys.argv) < 3:
    print("Please select input and output file")
    sys.exit()

in_file = codecs.open(sys.argv[1], encoding="utf-8")
out_file = codecs.open(sys.argv[2], "w", encoding="utf-8")

for line in in_file:
    text = json.loads(line)["text"]
    labels = json.loads(line)["labels"]
    for tag in labels:
        out_file.write(text[tag[0]:tag[1]] + "\t" + tag[2] + "\n")
    out_file.write("\n")
    

