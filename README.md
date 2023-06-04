# My Changes to privateGPT

This is my fork of [imartinez/privateGPT](https://github.com/imartinez/privateGPT) for playing around with my ideas for changes/improvements.

## Clarifications - Setup

***Before installing anything*** - `conda create` a virtual environment and use the python-version which (I hope) works with all the requirements.

```shell
conda create -n privategpt python=3.10.9
conda activate privategpt
```

Next, on M1/M2 Macs install pytorch,...

```shell
pip3 install torch torchvision torchaudio
```

continue with the installation according to in the main branch

```shell
pip3 install -r requirements.txt
```

Don't forget to create the `.env` file (e.g. by copying `example.env`) and then maybe tweak it (e.g. temperature)!&nbsp;

Please remember to use `conda activate` ... whenever you use privateGPT later!

## Changes Implemented

***for `privateGPT.py` (and `.env`):***

+ moved all commandline parameters to the `.env` file, no more commandline parameter parsing
  + `MUTE_STREAM`, `HIDE_SOURCE`
+ added LLM temperature parameter to `.env`
  + `MODEL_TEMP` with default 0.8 - I use .4 in `example.env` to reduce halucinations
+ refined `.env` sources parameter into filename and details
  + `HIDE_SOURCE` for everything and `HIDE_SOURCE_DETAILS` to have filenames only or also the details
+ added `.env` parameter for langchain-debugging
  + `LANGCHAIN_DEBUG`, default False
+ Tweaks for trying to avoid halucinations, if privateGPT finds nothing of relevance in the source_documents
  + used prompting technique suggested by [Guillaume-Fgt](https://github.com/Guillaume-Fgt) in [issue#517](https://github.com/imartinez/privateGPT/issues/517)
+ tried to polish the output-readability a bit, including base LLM callbacks (needs more work)

***for `ingest.py` (and `.env`):***

+ tweaked `ingest.py` and `.env` for changeable ingest-chunking
  + `CHUNK_SIZE`, default 500 and `CHUNK_OVERLAP`, default 50
+ re-enabled `ingest.py` for reading `.docx` (imartinez commented it out)
+ `ingestSpaCy.py` is an improved version of `ingest.py` for improved chunck quality
  + needs additional one-time installs before use:
    + pip install spacy
    + python -m spacy download en_core_web_sm
  + uses [spaCy](https://spacy.io/) for chunk-splitting
  + removes single \n, and replaces remaining multiple \n with single \n

## Ideas not yet implemented

+ make llama.cpp work (need to fix GGML version issues)
+ GPT4all yields much worse result than chatGPT
  + test llama.cpp models
  + maybe enable cloud chatGPT via configuration
+ output needs improvements (robustnes, readability)
  + reponse-streaming issues from LLM needs a fix, e.g. gpt4all is low-level streaming via print, causing aborts - this needs to move to more robust langchain streaming
  + better display of sources+their details after response
&nbsp;

&nbsp;

---------- the original `imartinez/privateGPT` README starts here ---------- &nbsp;

# privateGPT
Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain), [GPT4All](https://github.com/nomic-ai/gpt4all), [LlamaCpp](https://github.com/ggerganov/llama.cpp), [Chroma](https://www.trychroma.com/) and [SentenceTransformers](https://www.sbert.net/).

<img width="902" alt="demo" src="https://user-images.githubusercontent.com/721666/236942256-985801c9-25b9-48ef-80be-3acbb4575164.png">

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

Then, download the LLM model and place it in a directory of your choice:
- LLM: default to [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin). If you prefer a different GPT4All-J compatible model, just download it and reference it in your `.env` file.

Rename `example.env` to `.env` and edit the variables appropriately.
```
MODEL_TYPE: supports LlamaCpp or GPT4All
PERSIST_DIRECTORY: is the folder you want your vectorstore in
MODEL_PATH: Path to your GPT4All or LlamaCpp supported LLM
MODEL_N_CTX: Maximum token limit for the LLM model
EMBEDDINGS_MODEL_NAME: SentenceTransformers embeddings model name (see https://www.sbert.net/docs/pretrained_models.html)
TARGET_SOURCE_CHUNKS: The amount of chunks (sources) that will be used to answer a question
```

Note: because of the way `langchain` loads the `SentenceTransformers` embeddings, the first time you run the script it will require internet connection to download the embeddings model itself.

## Test dataset
This repo uses a [state of the union transcript](https://github.com/imartinez/privateGPT/blob/main/source_documents/state_of_the_union.txt) as an example.

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run privateGPT.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

Note: during the ingest process no data leaves your local environment. You could ingest without an internet connection, except for the first time you run the ingest script, when the embeddings model is downloaded.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python privateGPT.py
```

And wait for the script to require your input.

```plaintext
> Enter a query:
```

Hit enter. You'll need to wait 20-30 seconds (depending on your machine) while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again.

Note: you could turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.


### CLI
The script also supports optional command-line arguments to modify its behavior. You can see a full list of these arguments by running the command ```python privateGPT.py --help``` in your terminal.


# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `HuggingFaceEmbeddings` (`SentenceTransformers`). It then stores the result in a local vector database using `Chroma` vector store.
- `privateGPT.py` uses a local LLM based on `GPT4All-J` or `LlamaCpp` to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- `GPT4All-J` wrapper was introduced in LangChain 0.0.162.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

### For Windows 10/11
To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
   * Universal Windows Platform development
   * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the `gcc` component.

## Mac Running Intel
When running a Mac with Intel hardware (not M1), you may run into _clang: error: the clang compiler does not support '-march=native'_ during pip install.

If so set your archflags during pip install. eg: _ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt_

# Disclaimer
This is a test project to validate the feasibility of a fully private solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. The models selection is not optimized for performance, but for privacy; but it is possible to use different models and vectorstores to improve performance.
