import os
import sys
import json

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

# equivalent Linux: "wc -l"
def line_count(filename):
    f = open(filename)                  
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read # loop optimization
    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)
    return lines
		


	