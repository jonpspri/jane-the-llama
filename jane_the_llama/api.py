from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

import asyncio
import logging
import uvicorn

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

from .settings import JaneSettings
from .types import Session

settings = JaneSettings()

TIMEOUT = 1000 * 60 * 60 * 24

#
#  The redis is going to be divided into namespaces:
#    "sessions" for core session data
#
redis = Redis(host=settings.redis_host,
                  port=settings.redis_port,
                  protocol=3,
                  decode_responses=True,
                  db=0,
                  )
responses = {}

#
#  Preallocate as much of the environment as we can to reduce load during
#  the chat cycle
#

#  TODO:  Model IDs should be configurable to avoid code migrations
llm = WatsonxLLM(
                model_id="meta-llama/llama-3-2-90b-vision-instruct",
                url="https://us-south.ml.cloud.ibm.com",
                apikey=settings.watsonx_apikey,
                project_id=settings.watsonx_project_id,
                max_new_tokens=200
                )
vector_store = MilvusVectorStore(
                uri=settings.milvus_url,
                token=settings.milvus_token,
                dim=384, overwrite=False)
index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=WatsonxEmbeddings(
                        model_id="ibm/slate-30m-english-rtrvr-v2",
                        url="https://us-south.ml.cloud.ibm.com",
                        apikey=settings.watsonx_apikey,
                        project_id=settings.watsonx_project_id,
                        truncate_input_tokens=512,
                        embed_batch_size=20
                    ))
chat_engines = {}

retriever = index.as_retriever(similarity_top_k=2)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Location"],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@app.post("/sessions", status_code=201)
def create_session(response: Response) -> Session:
    session = Session()
    redis.setex(session.id, TIMEOUT, session.model_dump_json())
    response.headers['Location']=f"/sessions/{session.id}"
    return session

@app.get("/sessions")
def read_sessions():
    return [ x for x in redis.scan_iter() ]

@app.get("/sessions/{session_id}")
def read_session(session_id: str):
    session_str = redis.get(session_id)
    if session_str:
        return Response(content=session_str, media_type="application/json")  # Already a JSON representation!
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

async def _chat( session_id: str, content: str ) -> ChatMessage:
    chat_response = chat_engines[session_id].chat(content)
    session = Session.model_validate_json(redis.get(session_id))
    chat_message = ChatMessage(MessageRole.ASSISTANT, chat_response.response)
    session.chat_history.append(chat_message)
    redis.setex(session.id, TIMEOUT, session.model_dump_json())
    return chat_message

@app.post("/sessions/{session_id}/chat_messages", status_code=201)
async def add_chat_message(session_id: str, u: ChatMessage, response: Response) -> ChatMessage:
    session = Session.model_validate_json(redis.get(session_id))
    new_index = len(session.chat_history)
    session.chat_history.append(u)
    redis.setex(session.id, TIMEOUT, session.model_dump_json())

    if u.role == MessageRole.USER:
        chat_engines[session_id] = index.as_chat_engine(
                    chat_mode='condense_plus_context',
                    memory = ChatMemoryBuffer.from_defaults( llm=llm, chat_history=session.chat_history),
                    llm=llm
                    )
        responses[ (session_id, new_index) ] = asyncio.create_task( chat_engines[session_id].achat(u.content) )

    response.headers['Location']=f"/sessions/{session_id}/chat_messages/{new_index}"
    return u

@app.get("/sessions/{session_id}/chat_messages")
def read_chat_messages(session_id: str) -> list[ChatMessage]:
    session = Session.model_validate_json(redis.get(session_id))
    return session.chat_history

@app.get("/sessions/{session_id}/chat_messages/{index}")
def read_chat_message(session_id: str, index: int) -> ChatMessage:
    session = Session.model_validate_json(redis.get(session_id))
    if index < 0 or index > len(session.chat_history):  # TODO Can I replace this with a try and index out of range?
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    return session.chat_history[index]

@app.get("/sessions/{session_id}/chat_messages/{index}/response")
async def read_chat_message_response(session_id: str, index: int) -> ChatMessage:
    if ( session_id, index ) not in responses:
        raise HTTPException(status_code=404, detail=f"Response index {index} not found")
    return await responses[( session_id, index )]

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

    #  Check whether Milvus is properly configured





