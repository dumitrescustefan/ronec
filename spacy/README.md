# Spacy tutorial

This is a tutorial that shows how RONEC is integrated with [Spacy command line interface](https://spacy.io/api/cli).

## Convert 

Firstly, you need to convert the RONEC CoNLL-UP to [Spacy's JSON CoNLL-U BIO format](https://spacy.io/api/annotation#json-input) using the
`convert_spacy.py` script. It creates two files to train and validate the model (`dev_ronec.json` and `train_ronec.json`, respectievly). To run the script use the following command:

```
python3 convert_spacy.py <ronec_conllup_path> <output_path>
```

## Train

To train a model, you must give as arguments the path to train and dev files created from running the previous convert script to 
the Spacy's cli. Also, remember to add the `-p ner` argument, to use only the named entity recognition functionality. For instance,
run the following command:

```
python3 -m spacy train ro <model_path> <ronec_train_path> <ronec_dev_path> -p ner
```

Additional information about Spacy's training configuration can be found at https://spacy.io/api/cli#train.

## Evaluate

To evaluate the model, you must give as arguments the path to dev file created by the `convert_spacy.py` and the path to the trained model.
For instance, run the following command:

``` 
python3 -m spacy evaluate <model_path> <ronec_eval_path>
```

It gives the following results on the basic model (Note: to obtain better results, you need to tune the hyperparameters of the the model):

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

Additional information about Spacy's evaluation configuration can be found at https://spacy.io/api/cli#evaluate.
