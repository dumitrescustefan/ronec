![Version%202.0](https://img.shields.io/badge/version-2.0-red)

# RONEC - the Romanian Named Entity Corpus - v2.0 

RONEC, at version 2.0, holds **12330** sentences with over **0.5M** tokens, annotated with **15** classes, to a total of **80.283** distinctly annotated entities. 

It is more than twice the size of the previous version. It contains all data from v1 and everything has been annotated from scratch to confrom to a single standard. For version 1, please see the ``ronec_v1`` folder. 

## Corpus details

The corpus has the following classes and distribution in the train/valid/test splits:

| Classes      	| Total  	    | Train  	|         	| Valid  	|         	| Test   	|         	|
|-------------	|:------:	    |:------:	|:-------:	|:------:	|:-------:	|:------:	|:-------:	|
|            	| #     	    | #     	| %     	| # 	    | % 	    | #     	| %     	|
| PERSON      	|  **26130** 	| 19167  	|  73.35  	|  2733  	|  10.46  	|  4230  	|  16.19  	|
| GPE         	|  **11103** 	|  8193  	|  73.79  	|  1182  	|  10.65  	|  1728  	|   15.56 	|
| LOC         	|  **2467**  	|  1824  	|  73.94  	|  270   	|  10.94  	|  373   	|  15.12  	|
| ORG         	|  **7880**  	|  5688  	|  72.18  	|   880  	|  11.17  	|  1312  	|  16.65  	|
| LANGUAGE    	|   **467**  	|   342  	|  73.23  	|   52   	|  11.13  	|   73   	|  15.63  	|
| NAT_REL_POL 	|  **4970**  	|  3673  	|  73.90  	|   516  	|  10.38  	|   781  	|  15.71  	|
| DATETIME    	|  **9614**  	|  6960  	|  72.39  	|  1029  	|   10.7  	|  1625  	|   16.9  	|
| PERIOD      	|  **1188**  	|   862  	|  72.56  	|   129  	|  10.86  	|   197  	|  16.58  	|
| QUANTITY    	|  **1588**  	|  1161  	|  73.11  	|   181  	|   11.4  	|   246  	|  15.49  	|
| MONEY       	|  **1424**  	|  1041  	|  73.10  	|   159  	|  11.17  	|   224  	|  15.73  	|
| NUMERIC     	|  **7735**  	|  5734  	|  74.13  	|   814  	|  10.52  	|  1187  	|  15.35  	|
| ORDINAL     	|  **1893**  	|  1377  	|   72.74 	|   212  	|   11.2  	|   304  	|  16.06  	|
| FACILITY    	|  **1126**  	|   840  	|   74.6  	|   113  	|  10.04  	|   173  	|  15.36  	|
| WORK_OF_ART 	|  **1596**  	|  1157  	|  72.49  	|   176  	|  11.03  	|   263  	|  16.48  	|
| EVENT       	|  **1102**  	|   826  	|  74.95  	|   107  	|   9.71  	|   169  	|  15.34  	|


## Format

The data is available as a train/valid/test split in ``data``, as json files. Each file is a list of instances, where an instance is a dictionary that contains the following keys:

```json
{
  "id": 10454,
  "tokens": ["Pentru", "a", "vizita", "locația", "care", "va", "fi", "pusă", "la", "dispoziția", "reprezentanților", "consiliilor", "județene", ",", "o", "delegație", "a", "U.N.C.J.R.", ",", "din", "care", "a", "făcut", "parte", "și", "dl", "Constantin", "Ostaficiuc", ",", "președintele", "C.J.T.", ",", "a", "fost", "prezentă", "la", "Bruxelles", ",", "între", "1-3", "martie", "."], 
  "ner_tags": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PERSON", "O", "O", "O", "O", "O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "B-PERSON", "I-PERSON", "I-PERSON", "I-PERSON", "I-PERSON", "B-ORG", "O", "O", "O", "O", "O", "B-GPE", "O", "B-PERIOD", "I-PERIOD", "I-PERIOD", "O"], 
  "ner_ids": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 0, 0, 0, 0, 0, 5, 0, 19, 20, 20, 0], 
  "space_after": [true, true, true, true, true, true, true, true, true, true, true, true, false, true, true, true, true, false, true, true, true, true, true, true, true, true, true, false, true, true, false, true, true, true, true, true, false, true, true, true, false, false]
}
```

The ``tokens`` are the words of the sentence. The ``ner_tags`` are the string tags assigned to each token, following the BIO2 format. For example, the span ``"între", "1-3", "martie"`` has three tokens, but is a single class ``PERIOD``, marked as ``"B-PERIOD", "I-PERIOD", "I-PERIOD"``. 
The ``ner_ids`` are the integer encoding of each tag, to be compatible with the standard and to be quickly used for model training. Note that each ``B``-starting tag is odd, and each ``I``-starting tag is even.
The ``space_after`` is used to help if there is a need to detokenize the dataset. A ``true`` value means that there is a space after the token on that respective position. 


## Authors
+ [Stefan Daniel Dumitrescu](https://www.linkedin.com/in/stefandumitrescu/)


## Acknowledgements
Big thanks to [termene.ro](https://termene.ro/) for carefully annotating the full expanded dataset. RONEC v2 would not have seen the light of day without them!


## Cite
Please consider citing the following [paper](https://arxiv.org/abs/1909.01247) as a thank you to the authors of the RONEC, even if it describes v1 of the corpus and you are using v2: 
```
Dumitrescu, Stefan Daniel, and Andrei-Marius Avram. "Introducing RONEC--the Romanian Named Entity Corpus." arXiv preprint arXiv:1909.01247 (2019).
```
or in .bibtex format:
```
@article{dumitrescu2019introducing,
  title={Introducing RONEC--the Romanian Named Entity Corpus},
  author={Dumitrescu, Stefan Daniel and Avram, Andrei-Marius},
  journal={arXiv preprint arXiv:1909.01247},
  year={2019}
}
```
