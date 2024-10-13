from config import ConfigurationSet, config_from_json
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from os import path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis
from typing import Optional
from uuid import uuid4

import asyncio
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class JaneSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='JANE_')

    milvus_host: str = 'localhost'
    milvus_port: int = 19530
    redis_host: str = 'localhost'
    redis_port: int = 6379

    @property
    def milvus_url(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

settings = JaneSettings()

script_dir = path.dirname(path.abspath(__file__))
config = ConfigurationSet(
        config_from_json(path.join(script_dir, './config.json'), read_from_file=True, ignore_missing_paths=True),
        config_from_json(path.join(script_dir, '../config.json'), read_from_file=True, ignore_missing_paths=True),
        config_from_json('/run/secrets/config', read_from_file=True, ignore_missing_paths=True),
        )

TIMEOUT = 1000 * 60 * 60 * 24

#
#  The redis is going to be divided into namespaces:
#    "sessions" for core session data
#
class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    chat_history: Optional[list[ChatMessage]] = []

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
llm = WatsonxLLM(
                model_id="meta-llama/llama-3-2-90b-vision-instruct",
                url="https://us-south.ml.cloud.ibm.com",
                apikey=config['watsonx_apikey'],
                project_id="6a51b5ec-ff0d-4cca-9f25-7561888cd9cd",
                max_new_tokens=200
                )
logger.warn(f"Connecting to Milvus at {settings.milvus_url} as {config['milvus_username']}")
vector_store = MilvusVectorStore(
                uri=settings.milvus_url,
                token=':'.join([
                     config['milvus_username'],
                     config['milvus_password']
                ]),
                dim=384, overwrite=False)
index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=WatsonxEmbeddings(
                        model_id="ibm/slate-30m-english-rtrvr-v2",
                        url="https://us-south.ml.cloud.ibm.com",
                        apikey=config['watsonx_apikey'],
                        project_id=config['watsonx_project_id'],
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







