# Create a vector database from PDF files in ./pdf folder
import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

PATH_TO_PDFS = './pdfs/'
PATH_TO_DATABASE = './chroma_db_nccn'
DEVICE = 'cuda:0' # Set to 'cpu' for non-Nvidia GPUs to disable Acceleration

loaders = []
for file in os.listdir(PATH_TO_PDFS):
    full_path_to_file = f"{PATH_TO_PDFS}{file}"
    if os.path.isfile(full_path_to_file):
        print(f'Add file "{full_path_to_file}" for processing')
        loaders.append(PyPDFLoader(full_path_to_file))

docs = []
for file in loaders:
    print(f'Processing "{file.source}"')
    docs.extend(file.load())

#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': DEVICE})
#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=PATH_TO_DATABASE)

print(vectorstore._collection.count())
