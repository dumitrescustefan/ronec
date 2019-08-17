# Spacy tutorial

This is a tutorial that shows how ROENC is integrated with [Spacy command line interface](https://spacy.io/api/cli).

## Convert 

Firstly, you need to convert the ROENC to [Spacy's JSON CoNLL-U BIO format](https://spacy.io/api/annotation#json-input) using the 
`spacy_convert.py` script. It creates three files to train, validate and evalute the model. To run the script use the following command: 

```
python3 convert_spacy.py <roenc_path> <output_path>
```

## Train

To train a model, you must give as arguments the path to train and dev files created from running the previous convert script to 
the Spacy's cli. Also, remember to specify `ent` to `-p` argument, to use only the named entity recognition functionality. For instance,
run the following command:

```
python3 -m spacy train ro <model_path> <roenc_train_path> <roenc_dev_path> -p ent
```

Additional information about Spacy's training configuration can be found at https://spacy.io/api/cli#train.

## Evaluate

To evaluate the model, you must give as arguments the path to eval file created from the convert script and the path to the trained model.
For instance, run the following command:

``` 
python3 -m spacy evaluate <model_path> <roenc_eval_path>
```

Additional information about Spacy's evaluation configuration can be found at https://spacy.io/api/cli#evaluate.
