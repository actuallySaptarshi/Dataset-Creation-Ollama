# Synthetic Dataset Creator with Ollama, Langchain & ChromaDB

This tool helps to create synthetic datasets locally, using Ollama.

## The workflow: 

This script uses a different approach towards conventional synthetic dataset creation.

Instead of generating the question & answers pair in one go, the questions and answers are generated seperately.

**Why this approach?**

This way, you can check the quality/type of questions and answers seperately. This method also seems to work best when creating synthetic dataset with smaller models. This results in a high quality, personalized dataset.

**Important Note**

If you want to create the dataset from multiple sources (pdfs), it is recommended to re-run this script for every single file and merge the csv in the end. This method is a bit more time consuming but it works.

## Use the script:

**It is recommended to use a python virtual env for running this script**

### Step 1: Install the requirements: 

* Install Ollama
* Install the required models using: 
    * `ollama pull gemma3:4b-it-q8_0`
    * `ollama pull embeddinggemma:latest`
* Inside a python venv: `pip install -r requirements.txt`


### Step 2: Generating questions:

* Change the 
    * `PDF_PATH`: Path to the source pdf file. 
    * `OUTPUT_CSV`: File name of the output csv questions set.
    * `prompt_template`(will not be required to change in most cases): System prompt for generating questions. *Changing this will effect the questions outputs* 

* Run the script: `python data_gen_chroma_questions.py`

### Step 3: Generating Answers:

* Change the
    * `FILE_PATH_URL`: Path to the source pdf file.
    * `INPUT_CSV`: Path to questions csv (Set this the same as `OUTPUT_CSV` of the generate questions script).
    * `OUTPUT_CSV`: The final output of the dataset with questions and answers.
    * `template`(will not be required to change in most cases): System prompt for generating answers. *Changing this will effect the answers outputs* 

* Run the script: `python data_gen_answers.py`

### Using it for training a model

The output contains a csv with the columns questions & answers. You can export this csv to any format of your choice eg. JSON, JSONL etc.



