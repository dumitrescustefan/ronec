# Using Spacy and RONEC

This section shows how to use a NER (Named Entity Recognizer) to perform entity detection in arbitrary text. Let's say we have a sentence like: "Popescu Ion a fost la Cluj.". With Spacy, we'll do something like this:

```
import spacy

nlp = spacy.load(<model>)
doc = nlp("Popescu Ion a fost la Cluj.")

for ent in doc.ents:
	print(ent.text, ent.start_char, ent.end_char, ent.label_)
```

and we should see an output like:

```
Popescu Ion 0 11 PERSON
Cluj 22 26 GPE
```

Depending on how we load ``<model>``, there are three ways to use Spacy and RONEC. 

## Option 1. Train your own model and load it into Spacy. (Difficulty: Easy)

Please see the [tutorial](https://github.com/dumitrescustefan/ronec/blob/master/spacy/train-local-model) here.

## Option 2. Download the pre-trained online model and load it into Spacy. (Difficulty: Easier)

Download ``ronec.zip`` from ``online-model`` and unzip it in your folder of choice. Let's say that the path to the unzipped model is ``\home\user\model-best``. 

When loading spacy, simply load as: ``spacy.load("\home\user\model-best")`` and you're all set up. 

## Option 3. Use Spacy directly. (Difficulty: Easiest)

We're working to push the trained model in Spacy's repositories, so we can just load the model out of the box. However, we're not quite there. We'll update this section as soon as possible. 
