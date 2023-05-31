import os


from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter, RecursiveCharacterTextSplitter, \
    TextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredEPubLoader, UnstructuredWordDocumentLoader, \
    UnstructuredFileLoader, DirectoryLoader
from langchain.document_loaders import DirectoryLoader

load_dotenv()
openai_api = os.environ.get("OPENAI_API_KEY")

chunk_size = 512
chunk_overlap = 32
persist_directory = './persist'
data_directory = './data'
local_embeddings = False

# if local_embeddings:
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     hf = HuggingFaceEmbeddings(model_name=model_name)


# check if directory exists
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


def get_embedding_model(local=False):
    """
    Get the embedding model
    :param local:
    :return:
    """
    return OpenAIEmbeddings()
    # if local:
    #     return instructor_embeddings
    # else:
    #     return OpenAIEmbeddings()


def get_loader(filename):
    """
    Get the appropriate loader for the file type
    :param filename:
    :return:
    """

    try:
        if filename.endswith('.txt'):
            return TextLoader(filename)
        elif filename.endswith('.pdf'):
            return PyPDFLoader(filename)
        elif filename.endswith('.epub'):
            return UnstructuredEPubLoader(filename)
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            return UnstructuredWordDocumentLoader(filename)
        else:
            return None
    except Exception as e:
        print(e)
        return None


def create_vector_index():
    """
    Create a vector index from a file
    :return:
    """
    print("Creating vector index")
    loader = DirectoryLoader(data_directory)
    print('files loading')
    documents = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, encoding_name="cl100k_base")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    print(f'Documents chunks: {len(docs)}')
    print(docs[0])

    try:
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
        vectordb.persist()
    except Exception as e:
        print(e)
        return {
            "status": "error",
            "message": "Error creating index",
            "error": str(e)
        }

    return {
        "status": "success",
        "message": "Index created successfully",
    }