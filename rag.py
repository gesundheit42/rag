from langchain import hub
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from datasets import Dataset
from langchain_ollama import ChatOllama

def get_embeddings(model_name):
    model_name = "dunzhang/stella_en_1.5B_v5"
    model_kwargs = {'device': 'cuda',
                    'trust_remote_code': True}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def get_rag_chain(docs):
    """Builds the rag_chain and returns it along with the retriever"""

    llm = ChatOllama(model="llama3.1", temperature=0)
    embeddings = get_embeddings(model_name="dunzhang/stella_en_1.5B_v5")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vectorstore.add_documents(documents=docs, ids=uuids)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    #### RETRIEVAL and GENERATION ####

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def inspect(state):
        """Print the state passed between Runnables in a langchain and pass it on"""
        print(state)
        return state

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RunnableLambda(inspect)
        | prompt
        | RunnableLambda(inspect)
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever, embeddings


def create_ragas_dataset(rag_chain, retriever, questions, ground_truths):
    answers = []
    contexts = []

    for i, query in enumerate(questions):
        if i > 4:
            break
        print(f"---DATASET CREATION: ANSWERING QUERY {i}/{len(questions)}")
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.invoke(query)])
  
    data = {
        "user_input": questions[:5],
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths[:5]
    }
    dataset = Dataset.from_dict(data)

    return dataset