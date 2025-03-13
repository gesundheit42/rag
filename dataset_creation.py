import os
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import get_rag_chain

pdf_directory = "papers"

all_documents = []

for filename in os.listdir(pdf_directory):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_directory, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(all_documents)

rag_chain, retriever, embeddings, llm_base = get_rag_chain(splits)

import json

from rag import create_query_construction_ragas_dataset


with open("/opt/cloudadm/llm_rag/benchmark.json", "r") as json_file:
    data = json.load(json_file)
    dataset = create_query_construction_ragas_dataset(llm_base, retriever, data['questions'], data['ground_truths'])
from datasets import load_from_disk

dataset.save_to_disk("./query_construction_dsd")