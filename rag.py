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
from langchain.prompts import ChatPromptTemplate
from operator import itemgetter

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
    #embeddings = get_embeddings(model_name="dunzhang/stella_en_1.5B_v5")
    embeddings = get_embeddings(model_name="Alibaba-NLP/gte-Qwen2-7B-instruct")
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
    rag_chain.invoke("What are the Rules for Propositional Connectives?")

    return rag_chain, retriever, embeddings, llm


def create_ragas_dataset(rag_chain, retriever, questions, ground_truths):
    answers = []
    contexts = []

    for i, query in enumerate(questions):
        print(f"---DATASET CREATION: ANSWERING QUERY {i + 1}/{len(questions)}")
        answers.append(rag_chain.invoke(query))
        contexts.append([docs.page_content for docs in retriever.invoke(query)])
  
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)

    return dataset

def query_construction(question, llm, retriever):
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple search queries related to: {question} \n
    Output (3 queries) and nothing else:"""
    prompt_decomposition = ChatPromptTemplate.from_template(template)
    generate_queries_decomposition = ( prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n")))
    question = "What are the Rules for Propositional Connectives?"
    questions = generate_queries_decomposition.invoke({"question":question})
    questions = generate_queries_decomposition.invoke({"question": question})

    def format_qa_pair(question, answer):
        """Format Q and A pair"""
        
        formatted_string = ""
        formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
        return formatted_string.strip()

    template = """Here is the question you need to answer:

    \n --- \n {question} \n --- \n

    Here is any available background question + answer pairs:

    \n --- \n {q_a_pairs} \n --- \n

    Here is additional context relevant to the question: 

    \n --- \n {context} \n --- \n

    Use the above context and any background question + answer pairs to answer the question: \n {question}
    """

    decomposition_prompt = ChatPromptTemplate.from_template(template)
    q_a_pairs = ""
    for q in questions:
        rag_chain = (
        {"context": itemgetter("question") | retriever, 
        "question": itemgetter("question"),
        "q_a_pairs": itemgetter("q_a_pairs")} 
        | decomposition_prompt
        | llm
        | StrOutputParser())

        answer = rag_chain.invoke({"question":q,"q_a_pairs":q_a_pairs})
        q_a_pair = format_qa_pair(q,answer)
        q_a_pairs = q_a_pairs + "\n---\n"+  q_a_pair

    return answer

def create_query_construction_ragas_dataset(llm, retriever, questions, ground_truths):
    answers = []
    contexts = []

    for i, query in enumerate(questions):
        print(f"---DATASET CREATION: ANSWERING QUERY {i + 1}/{len(questions)}")
        answers.append(query_construction(query, llm, retriever))
        contexts.append([docs.page_content for docs in retriever.invoke(query)])
  
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)

    return dataset

def create_base_dataset(llm, questions, ground_truths):
    answers = []
    parser = StrOutputParser()
    contexts = []
    for i, query in enumerate(questions):
        print(f"---DATASET CREATION: ANSWERING QUERY {i}/{len(questions)}")
        answers.append(parser.parse(llm.invoke(query).content))
        contexts.append([])
  
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)

    return dataset