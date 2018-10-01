# -*- coding: utf-8 -*-

# core data format
from functools import total_ordering

@total_ordering
class Annotation (object):
    def __init__ (self, start=-1, stop=-1, type=""): # stop is inclusive, e.g.: "x" is 0,1 while "xx" is 0,2
        self.start = start
        self.stop = stop
        self.type = type
      
    def cmp(self, other):
        if self.start != other.start:
            return -1 if self.start < other.start else 1
        elif self.stop != other.stop:
            return -1 if self.stop < other.stop else 1
        elif self.type != other.type:
            return -1 if self.type < other.type else 1
        else:
            return 0
            
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            #return self.__dict__ == other.__dict__
            return True if self.cmp(other) == 0 else False
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def __lt__(self, other):
        return True if self.cmp(other) == -1 else False
        
    def __gt__(self, other):
        return True if self.cmp(other) == 1 else False    

    def __le__(self, other):
        return True if self.cmp(other) != 1 else False  

    def __ge__(self, other):
        return True if self.cmp(other) != -1 else False            
    
    def __cmp__(self,other):                    
        return self.cmp(other)
    
    def __repr__(self):
        return "{{{},{}-{}}}".format(self.type, self.start, self.stop)
    
    def __hash__(self):
        return id(self.type)+id(self.start)+id(self.stop)    

class Sentence (object):
    def __init__ (self, sentence, annotations = []):
        self.sentence = sentence
        self.annotations = annotations # list of lists of Annotation objects each list is a different annotation layer as a sentence can have several different annotations
        self.annotations.sort(reverse=False)
        
    def __repr__(self):
        #print("\n\033[44m"+self.sentence+"\033[0m")
        line = ""
        line+="\n"+self.sentence+"\n"
        for ann in self.annotations:
            text = self.sentence[ann.start:ann.stop]
            line+="\t{}  {}-{}\t{}\n".format(text.rjust(40), str(ann.start).rjust(3), str(ann.stop).ljust(3), ann.type)
        return line
        
        
def process_split_exceptions (original_sentence_object):
    from dtw import dtw
    import copy
    
    def custom_distance(x, y):
        return 0. if x==y else 1.
        
    def first_index (list, value):
        for index, elem in enumerate(list):
            if elem == value:
                return index
        return -1
        
    def last_index (list, value):
        for index, elem in enumerate(list[::-1]):
            if elem == value:
                return len(list) - index - 1
        return -1    
        
    def rindex(mylist, myvalue):
        return len(mylist) - mylist[::-1].index(myvalue) - 1
        
   
    sentence = copy.deepcopy(original_sentence_object.sentence)
    annotations = copy.deepcopy(original_sentence_object.annotations)
    annotations.sort(reverse=False)
    new_sentence_object = copy.deepcopy(original_sentence_object) 
    
    while(True):
        # search for joined annotations              
        found_join = False
        for i in range(len(annotations)-1):
            ann_left = annotations[i]
            ann_right = annotations[i+1]
                        
            #print("i = "+str(i))
            #print(ann_left)
            #print(ann_right)
            if ann_left.stop == ann_right.start - 1 and sentence[ann_left.stop] != ' ':
                found_join = True
                #print("FOUND JOINED ANNOTATIONS!")
                
                # split here                
                new_sentence = sentence[:ann_left.stop]+" "+sentence[ann_left.stop]+" "+sentence[ann_left.stop+1:]
                #print(" NEW SENT: "+new_sentence)
                
                # redo annotations
                dist, cost, acc, path = dtw(sentence, new_sentence, dist=custom_distance)
                new_annotations = []
                #print(path[0])
                #print(path[1])
                for annotation in annotations:
                    #print(annotation)
                    new_start = first_index(path[0], annotation.start)
                    new_stop = first_index(path[0], annotation.stop-1) + 1        
                    new_annotation = Annotation(new_start, new_stop, annotation.type)
                    new_annotations.append(new_annotation)
                    #print(new_annotation)      
                annotations = new_annotations
                sentence = new_sentence
                
                new_sentence_object = Sentence(new_sentence, new_annotations)
                break
                
        if not found_join:
            break
            
    return new_sentence_object
        
def process_split_exceptions_old (sentence):
    """
    numar-numar => numar - numar
    numar/numar => numar / numar
    cuvant“ => cuvant “
    cuvant-cuvant => cuvant - cuvant
    
    regex:
    line = re.sub(r'([0-9]+)-([0-9]+)', r'\1 - \2', line)
    line = re.sub(r'([0-9]+)/([0-9]+)', r'\1 / \2', line)
    line = re.sub(r'(\w+)“', r'\1 “', line, re.UNICODE)
    line = re.sub(r'(\w+)-(\w+)', r'\1 - \2', line, re.UNICODE)
    """
    from dtw import dtw
    import re, copy
    
    def custom_distance(x, y):
        return 0. if x==y else 1.
    
    def first_index (list, value):
        for index, elem in enumerate(list):
            if elem == value:
                return index
        return -1
        
    def last_index (list, value):
        for index, elem in enumerate(list[::-1]):
            if elem == value:
                return len(list) - index - 1
        return -1    
        
    def rindex(mylist, myvalue):
        return len(mylist) - mylist[::-1].index(myvalue) - 1
        
    new_sent = copy.deepcopy(sentence.sentence)
    new_sent = re.sub(r'([0-9]+)-([0-9]+)', r'\1 - \2', new_sent)
    new_sent = re.sub(r'([0-9]+)/([0-9]+)', r'\1 / \2', new_sent)
    new_sent = re.sub(r'(\w+)“', r'\1 “', new_sent, re.UNICODE)
    new_sent = re.sub(r'(\w+)-(\w+)', r'\1 - \2', new_sent, re.UNICODE)
        
    #print(sentence.sentence)
    #print(new_sent)
    dist, cost, acc, path = dtw(sentence.sentence, new_sent, dist=custom_distance)
    #print(path[0])
    #print(path[1])
    
    new_annotations = []
    for annotation in sentence.annotations:
        #print(annotation)
        new_start = last_index(path[0], annotation.start)
        new_stop = first_index(path[0], annotation.stop-1) + 1        
        new_annotation = Annotation(new_start, new_stop, annotation.type)
        new_annotations.append(new_annotation)
        #print(new_annotation)        
    return Sentence(new_sent, new_annotations)

"""    
#test    
annotations = []
annotations.append(Annotation(0,2,"X"))
annotations.append(Annotation(3,5,"Y"))
annotations.append(Annotation(6,13,"Z"))
annotations.append(Annotation(14,18,"Q"))
s = Sentence("12-34 Ana Buc-Deva",annotations)
print(s)    
new_s = process_split_exceptions(s)    
print("_"*20)
print(new_s)
"""  