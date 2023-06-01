import hashlib
import os
from datetime import datetime

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from vectordb.vectordb import create_vector_index
from conversation.conv import get_response_over_doc, get_simple_response

app = FastAPI()
debug = True

origins = [
    "http://localhost.com",
    "https://localhost.com",
    "http://localhost",
    "http://localhost:3000",
    "http://sokaris.link:3000",
    "http://sokaris.link",
    "https://assistant.sokaris.link",
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConvRequest(BaseModel):
    user_message: str           # user message
    history: list = None        # history of the conversation
    type: int = None            # type of the conversation (for the prompt)
    data_id: int = 1            # id of the document
    api_key: str = None         # api key


def check_api_key(api_key):
    LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
    # md5_hash = hashlib.md5((LOCAL_API_KEY + datetime.now().strftime("%Y-%m-%d")).encode()).hexdigest()
    md5_hash = hashlib.md5((LOCAL_API_KEY).encode()).hexdigest()

    if api_key == md5_hash:
        return True
    else:
        return False



@app.get("/")
def read_root():
    return {"Page not found"}


# CONVERSATIONS
@app.post("/conv/get_response")
async def get_response(body: ConvRequest):

    if not check_api_key(body.api_key):
        return {
            "response": "Wrong API key",
            "code": 400
        }

    resp = get_simple_response(body.user_message, body.history, body.type)

    return {
        "response": resp,
        "code": 200
    }


@app.post("/conv/get_response_doc")
async def get_response_doc(body: ConvRequest):

    if not check_api_key(body.api_key):
        return {
            "response": "Wrong API key",
            "code": 400
        }

    resp = get_response_over_doc(body.user_message, body.history, body.type, body.data_id)

    return {
        "response": resp,
        "code": 200
    }


# @app.post("/docs/uploadfile/")
# async def create_upload_file():
#
#     res = create_vector_index()
#     print('Uploading')
#
#     return {
#         "result": res,
#         "code": 400,
#     }
