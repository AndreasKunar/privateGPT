#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain import PromptTemplate
import os

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_temp = os.environ.get('MODEL_TEMP',0.8)
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
mute_stream = os.environ.get('MUTE_STREAM',"False") != "False"
hide_source = os.environ.get('HIDE_SOURCE',"False") != "False"
hide_source_details = os.environ.get('HIDE_SOURCE_DETAILS',"False") != "False"

from constants import CHROMA_SETTINGS

def main():
    # moved all command line arguments to .env
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False, temperature=model_temp)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False, temp=model_temp)
        case _default:
            print(f"Model {model_type} not supported!")
            exit
            
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not hide_source,
        chain_type_kwargs=chain_type_kwargs,
    )
    # Interactive questions and answers
    while True:
        query = input("\n### Enter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain and stream-print the result
        print("\n### Answer by privateGPT via "+os.path.split(model_path)[1]+" with temperature:"+model_temp+":")
        res = qa(query)
        answer, docs = res['result'], [] if hide_source else res['source_documents']

        # Print the relevant sources used for the answer
        if docs:
            print("--- Sources: ----")
        for document in docs:
            print(">>" + os.path.split(document.metadata["source"])[1]+":")
            if not hide_source_details:
                print(document.page_content)
        print("-------------------")

if __name__ == "__main__":
    main()
