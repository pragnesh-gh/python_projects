## 1. Ingest PDF Files
# 2. Extract Text from PDF Files and split into small chunks
# 3. Send the chunks to the embedding model
# 4. Save the embeddings to a vector database
# 5. Perform similarity search on the vector database to find similar documents
# 6. retrieve the similar documents and present them to the user
## run pip install -r requirements.txt to install the required packages


#================================Loading the required documents================================
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OnlinePDFLoader

doc_path = "data/DL_LectureSlides.pdf"
model = "gemma3:4b"


#Local PDF uploads
if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("done loading ...")
else:
    print("Upload a PDF file.")

#Preview the first page
content = data[0].page_content
# print(content[:100])

#================================Splitting into small chunks================================

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1200, chunk_overlap = 300)
chunks = text_splitter.split_documents(data)
print("done splitting ...")

print(f"Number of chunks: {len(chunks)}")
print(f"Example chunk: {chunks[0]}")

#================================Add to Vector Database================================
#First we need to make embedding s then add to the Vector database
import ollama
ollama.pull("nomic-embed-text")

#creating the vector database
vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="simple-rag",
)
print("done adding to vector database ...")


#================================Retrieval================================
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser #A way to parse whatever comes in or goes through the RAG system

from langchain_ollama import ChatOllama #used as our LLM

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

#setting up the model
llm = ChatOllama(model=model) #helps in generating multiple questions from one single question and use that to guide the retrieval process.

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five different versions of 
    the given user question to retrieve relevant documents from a vector database. By generating multiple 
    perspectives on the user question, your goal is to help the user overcome some of the limitations of 
    the distance-based similarity search. Provide these alternative questions separated by newlines.
    Original question: {question} """,
)

retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(),llm,prompt=QUERY_PROMPT)
# we are transforming the vector_db here into a retriever so that actually retrieve stuff.

#RAG prompt
template = """ Answer the question based ONLY on the following content: {content}
Question: {question}
"""
#internally the langchain funcitons know how to pull in the question and the context.

prompt = ChatPromptTemplate.from_template(template)


#Putting everything into a chain that actually goes through the whole process and puts everything together
chain = (
    {"content": retriever, "question" : RunnablePassthrough()}
    | prompt.bind()
    | llm
    | StrOutputParser()
)

res = chain.invoke({"question": "What is the document about?"})
print(res)






