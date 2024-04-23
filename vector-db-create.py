# Create a vector database from PDF files in ./pdf folder
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document


PATH_TO_PDFS = './pdfs/'
PATH_TO_DATABASE = './chroma_db_nccn'
DEVICE = 'cuda:0' # Set to 'cpu' for non-Nvidia GPUs to disable Acceleration

embedding_function : HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': DEVICE})
for file in os.listdir(PATH_TO_PDFS):
    full_path_to_file = f"{PATH_TO_PDFS}{file}"
    if os.path.isfile(full_path_to_file):
        print(f'Processing file "{file}"')

        print('  - Preparing loader...')
        loader : PyPDFLoader = PyPDFLoader(full_path_to_file)
        docs = loader.load()

        print('  - Loading text...')
        full_document_text : str = " ".join([doc.page_content for doc in docs])
        full_document_metadata = {
                "filename": file
            }
        full_document : Document = Document(page_content=full_document_text, metadata=full_document_metadata)

        print('  - Splitting document...')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[ "\n\n", "\n" ])
        splitted_doc = text_splitter.split_documents( [ full_document ])
        
        print('  - Creating embbedings...')
        Chroma.from_documents(documents=splitted_doc, embedding=embedding_function, persist_directory=PATH_TO_DATABASE)

print('End')
