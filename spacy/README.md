# Spacy tutorial

This is a demo tutorial that shows how RONEC is integrated with [Spacy command line interface](https://spacy.io/api/cli).

## Convert 

Firstly, you need to convert the RONEC CoNLL-UP to [Spacy's JSON CoNLL-U BIO format](https://spacy.io/api/annotation#json-input) using the
`convert_spacy.py` script. It creates two files to train and validate the model (`train_ronec.json` and `dev_ronec.json`, respectively) to the specified path.

```
python3 convert_spacy.py [ronec_conllup_path] [output_path] [validation_ratio]
```

| Argument | Type | Description |
| --- | --- | --- |
| ronec_conllup_path | str | Path of the ronec CoNLL-U Plus file. |
| output_path | str | Save path of the train and dev files in Spacy's JSON CoNLL-U BIO format. |
| --dev_ratio | float | Fraction of the training data to be used as dev data. Default: `0.1` |

## Train

To train a model, you must give as arguments the path to train and dev files created from running the previous convert script to 
the Spacy's cli. Also, remember to add the `-p ner` argument, to use only the named entity recognition functionality.

```
python3 -m spacy train ro [model_path] [ronec_train_path] [ronec_dev_path] -p ner
```

| Argument | Type | Description |
| --- | --- | --- |
| model_path | str | Path of the model. |
| ronec_train_path | str | Path of the train file in Spacy's JSON CoNLL-U BIO format. |
| ronec_dev_path | str | Path of the dev file in Spacy's JSON CoNLL-U BIO format. |

Additional information about Spacy's training configuration can be found at https://spacy.io/api/cli#train.

## Evaluate

To evaluate the model, you must give as arguments the path to dev file created by the `convert_spacy.py` and the path to the trained model.

``` 
python3 -m spacy evaluate [model_path] [ronec_test_path]
```

| Argument | Type | Description |
| --- | --- | --- |
| model_path | str | Path of the model. |
| ronec_test_path | str | Path of the test file in Spacy's JSON CoNLL-U BIO format. |


It gives the following results on the default model:

```
Time      1.02 s
Words     18171
Words/s   17737
TOK       100.00
POS       0.00
UAS       3.05
LAS       0.00
NER P     83.69
NER R     80.62
NER F     82.13
```

Note: To obtain better results, you need to tune the hyperparameters of the the model

Additional information about Spacy's evaluation configuration can be found at https://spacy.io/api/cli#evaluate.

## Spacy API

The following code shows how to load and run a trained model to extract named entities from Romanian texts with the Spacy API.

```
import spacy

nlp = spacy.load(<model_path>)
doc = nlp("Popescu Ion a fost la Cluj")

for ent in doc.ents:
	print(ent.text, ent.start_char, ent.end_char, ent.label_)
```

Outputs:

```
Popescu Ion 0 11 PERSON
Cluj 22 26 GPE
```
