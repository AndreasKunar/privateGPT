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

Continue with the installation according to in the main branch

```shell
pip3 install -r requirements.txt
```

***For using lama-models:*** update llama-cpp-python to latest version (for latest model-compatibility)

```shell
pip3 install --update llama-cpp-python   
```

... and use the models from [here](https://huggingface.co/TheBloke), also change the `MODEL_PATH` accordingly, as well as `MODEL_TYPE=LlamaCpp` in `.env`
&nbsp;

Create/tweak the `.env` file (e.g. by copying `example.env`) for any non-default parameter-settings
&nbsp;

Please remember to use `conda activate` ... whenever you use privateGPT later!

## Changes Implemented

***for `privateGPT.py` (and `.env`):***

+ moved all commandline parameters to the `.env` file, no more commandline parameter parsing
+ removed `MUTE_STREAM`, always using streaming for generating response
+ added LLM temperature parameter to `.env`
  + `MODEL_TEMP` with default 0.8 - I use .4 in `example.env` to reduce halucinations
+ refined sources parameter (initially commandlineno in `.env`) into filename and details
  + `HIDE_SOURCE` for everything and `HIDE_SOURCE_DETAILS` to have filenames only or also the details
+ added `.env` parameter for langchain-debugging
  + `LANGCHAIN_DEBUG`, default False
+ Tweaks for trying to avoid halucinations, if privateGPT finds nothing of relevance in the source_documents
  + used prompting technique suggested by [Guillaume-Fgt](https://github.com/Guillaume-Fgt) in [issue#517](https://github.com/imartinez/privateGPT/issues/517)
+ tried to polish the output-readability a bit, including base LLM callbacks (needs more work)
+ tweaks for making `LlamaCpp` work - Note: llama_cpp/llama.py does NOT run in the Python debugger, aborts loading!

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

+ GPT4all yields much worse result than chatGPT
  + evaluating if/which llama.cpp model is better, maybe tweakig prompting
  + maybe enable cloud chatGPT via configuration
+ output improvements - better display of sources+their details after response
+ maybe gradio webUI
&nbsp;

## Please continue with the original [imartinez/privateGPT/README](https://github.com/imartinez/privateGPT/blob/main/README.md) for more
