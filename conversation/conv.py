import os

from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


from vectordb.vectordb import get_embedding_model

load_dotenv()

llm = ChatOpenAI(temperature=.0, model_name="gpt-3.5-turbo", verbose=True)

persist_directory = './persist'


def format_response(response_input):
    """
    Format the response
    :param response_input:
    :return:
    """
    print("Response input: ", response_input)
    return response_input


def get_simple_response(prompt, history, type):
    """
    Get a simple response from the model
    :param prompt:
    :param history:
    :param type:
    :return:
    """

    # system = PromptTemplate(
    #     template="You are a helpful assistant that translates {input_language} to {output_language}.",
    #     input_variables=["input_language", "output_language"],
    # )
    system = PromptTemplate(
        template="",
        input_variables=[],
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=system)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    try:
        response = chain.run(text=prompt)
    except Exception as e:
        print("Error in getting response: ", e)
        return {
            "status": "error",
            "message": str(e),
        }

    return {
        "status": "success",
        "message": response,
    }


def get_response_over_doc(prompt, history, type, data_id):

    embeddings = get_embedding_model()
    data_dir = os.path.join(persist_directory, str(data_id))
    docsearch = Chroma(persist_directory=data_dir, embedding_function=embeddings)

    _DEFAULT_TEMPLATE = """Given the context information answer the following question.
    Answer in a language of the question.
    If you don't know the answer, just say you dont know Don't try to make up an answer.
    =========
    question: {}""".format(prompt)

    retriever = docsearch.as_retriever(enable_limit=True, limit=5, search_kwargs={"k": 3})

    cur_conversation = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    try:
        result = cur_conversation({"query": _DEFAULT_TEMPLATE})
        response = format_response(result['result'])

    except Exception as e:
        print("Error in getting response: ", e)
        return {
                "status": "error",
                "message": "Error while getting response",
                "data": {
                    "response": str(e),
                }
            }

    return {
        "status": "success",
        "message": "Agent response",
        "data": {
            "response": response,
            "source": result["source_documents"],
        }
    }
