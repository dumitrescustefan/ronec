# Doccano: Text Annotation Tool

## What is Doccano?

[Doccano](https://github.com/chakki-works/doccano) is one of the best open source tools for text annotations. 

## Get Started

In order to get started, Doccano needs to be hosted on a server that can be accessed remotely. There are two recommended ways to use Doccano:
- [Docker Compose]
- [Docker]

It is recomended to use Docker Compose, but because the current version has some limitations  with the imported file size, we will use the Docker version.

## Docker

Create a Docker container for Doccano:

```bash
docker pull doccano/doccano
docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=password" \
  -p 8000:8000 doccano/doccano
```

Next, start Doccano by running the container:

```bash
docker container start doccano
```

To stop the container, run:

```bash
docker container stop doccano
```
All datasets in the container will be saved.

Go to <http://127.0.0.1:8000/>.

## Create project
We need to create a new project for annotation. After login with selected credentials, you will need to nevigate to main project list page of Doccano.
There you will crete a new project by clicking `Create Project` button and select project type as `sequence labeling`. Add a name with some descriptions 
for the project.

## Import data
After creating the project, we will need to import the data. Since Doccano suports only one label per word at a time, we need to extract only two columns and feed them into `CONLL` format.
Download the dataset in **IOB format (~14.5MB): [IOB Download](https://github.com/dumitrescustefan/ronec/raw/master/ronec/conllup/zips/ronec_iob.zip)** or
**PARSEME:MWE format (~14.3MB): [PARSEME:MWE Download](https://github.com/dumitrescustefan/ronec/raw/master/ronec/conllup/zips/ronec.zip)** and use `/scripts/conllu2conll.py` to extract the columns:
```bash
conllu2conll.py input_corpus output_corpus column_number
```
`output_corpus` will contain the word and the `column_number` label from dataset. By default it will select the last column.
In the `Import Data` page, select `CoNLL` format. Click `Select a file` button. Select `output_corpus` and it will be loaded automatially.
After importing the dataset and the existing labels will be visible.

## Define labels

Click `Labels` button to define new labels. Select text, colors and shortcut keys for each label. Labels respect a format and a standard. More information can be found here [CONLLUP](http://universaldependencies.org/ext-format.html).

## Annotate
Ok, now we are ready to annotate. Click `Annotate Data` button and afterwards we can start annotating by selecting text and use the shortcut key to define the labels.

## Export data

After annotation, we can download the data. Click `Edit Data` button and `Export Data`. Select `JSON` format for download. 
In order to convert to `CONLL` format use `/scripts/json2conll.py`:
```bash
json2conll.py json_corpus conll_corpus
```
The `conll_corpus` will contain two columns, with word and label.

Congratulations! You have mastered how to use Doccano for a sequence labeling project.