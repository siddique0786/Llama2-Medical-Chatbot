# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# DATA_PATH = 'data/'
# DB_FAISS_PATH = 'vectorstore1/db_faiss1'
#
# # Create vector database
# def create_vector_db():
#     loader = DirectoryLoader(DATA_PATH,
#                              glob='*.pdf',
#                              loader_cls=PyPDFLoader)
#
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
#                                                    chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)
#
#     embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
#                                        model_kwargs={'device': 'cpu'})
#
#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local(DB_FAISS_PATH)
#
# if __name__ == "__main__":
#     create_vector_db()


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore1/db_faiss1'


# Create vector database
def create_vector_db():
    if not os.path.exists(DATA_PATH):
        print(f"Data path '{DATA_PATH}' does not exist.")
        return

    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("No documents found. Ensure there are PDF files in the specified directory.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database saved at '{DB_FAISS_PATH}'")


if __name__ == "__main__":
    create_vector_db()

