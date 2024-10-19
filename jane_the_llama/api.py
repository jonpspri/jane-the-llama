import uvicorn

from asyncio import Lock, Task, create_task
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from loguru import logger
from redis import Redis
from typing import Self
from uuid import uuid4
from weakref import WeakValueDictionary

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import AgentChatResponse, ChatMode
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

from .settings import JaneSettings
from .types import Session

settings = JaneSettings()
TIMEOUT = 1000 * 60 * 60 * 24

redis = Redis(host=settings.redis_host,
                  port=settings.redis_port,
                  protocol=3,
                  decode_responses=True,
                  db=0,
                  )

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
                max_new_tokens=200,
                )
vector_store = MilvusVectorStore(
                uri=settings.milvus_url,
                token=settings.milvus_token,
                dim=384,
                overwrite=False,
                )
vector_index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=WatsonxEmbeddings(
                        model_id="ibm/slate-30m-english-rtrvr-v2",
                        url="https://us-south.ml.cloud.ibm.com",
                        apikey=settings.watsonx_apikey,
                        project_id=settings.watsonx_project_id,
                        truncate_input_tokens=512,
                        embed_batch_size=20,
                    ))
retriever = vector_index.as_retriever(similarity_top_k=3)

class SessionIdLockDict(dict):
    def __missing__(self, key):
        self.data[key] = Lock()

class SessionImpl:

    _instances: WeakValueDictionary[str, Self] = WeakValueDictionary()
    _tasks: dict[str, Task] = {}

    @classmethod
    def four(cls, session: Session):
        if session.id in cls._instances:
            impl = cls._instances[session.id]
            impl.session = session
        return SessionImpl(session)

    @classmethod
    def from_redis(cls, session_id: str):
        session_str = redis.get(session_id)
        assert isinstance(session_str, str)
        session = Session.model_validate_json(session_str)
        return cls.four(session)

    def __init__(self, session: Session):
        self.session = session
        self.__class__._instances[session.id] = self
        self.chat_engine = vector_index.as_chat_engine(
                    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
                    llm=llm,
                    )
        self.responses : dict[int, Task] = {}
        self.lock = Lock()

    @property
    def id(self):
        return self.session.id

    @property
    def chat_history(self):
        return self.session.chat_history

session_locks = dict[str, Lock]

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

@app.post("/sessions/{session_id}/chat_messages", status_code=201)
async def add_chat_message(session_id: str, u: ChatMessage, response: Response) -> ChatMessage:
    logger.debug(f"Received new message {u}")
    session_impl = SessionImpl.from_redis(session_id)
    async with session_impl.lock:
        session_impl = SessionImpl.from_redis(session_id)
        assert session_impl.chat_history is not None
        i = len(session_impl.chat_history)

        if u.role == MessageRole.USER:
            logger.debug(f"Chat Engine to start for ( {session_id}, {i} )")
            # TODO:  Use a method in SessionImpl to manage this
            task_id = str(uuid4())
            a_chat_history = session_impl.chat_history.copy()
            SessionImpl._tasks[task_id] = create_task(
                    session_impl.chat_engine.achat(u.content, a_chat_history)
                    )
            u.additional_kwargs["response_task"] = task_id
            logger.debug(f"Chat Engine started for task {task_id}")

        session_impl.chat_history.append(u)
        redis.setex(session_impl.id, TIMEOUT, session_impl.session.model_dump_json())

    message_uri=f"/sessions/{session_id}/chat_messages/{i}"
    response.headers['Location']=message_uri
    logger.debug(f"Returning message {message_uri}")
    return u

@app.get("/sessions/{session_id}/chat_messages")
def read_chat_messages(session_id: str) -> list[ChatMessage]:
    session_impl = SessionImpl.from_redis(session_id)
    return session_impl.chat_history

@app.get("/sessions/{session_id}/chat_messages/{index}")
def read_chat_message(session_id: str, index: int) -> ChatMessage:
    session_impl = SessionImpl.from_redis(session_id)

    # TODO Can I replace this with a try and index out of range?
    if index < 0 or index > len(session_impl.chat_history):
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    return session_impl.chat_history[index]

@app.get("/sessions/{session_id}/chat_messages/{index}/response",
         response_class=PlainTextResponse,
         )
async def read_chat_message_response(session_id: str, index: int, response: Response) -> str:
    session_impl = SessionImpl.from_redis(session_id)
    task_id = session_impl.chat_history[ index ].additional_kwargs[ "response_task" ]
    logger.debug(f"Waiting for a reply for task {task_id}")
    agent_chat_response = await SessionImpl._tasks[ task_id ]

    async with session_impl.lock:
        session_impl = SessionImpl.from_redis(session_id)
        assert isinstance(agent_chat_response, AgentChatResponse)
        cm = ChatMessage( role=MessageRole.ASSISTANT, content=str(agent_chat_response) )
        cm.additional_kwargs['response_id'] = task_id
        new_index = len(session_impl.chat_history)
        session_impl.chat_history.append(cm)
        redis.setex(session_impl.id, TIMEOUT, session_impl.session.model_dump_json())

    response.headers['Location']=f"/sessions/{session_impl.id}/chat_messages/{new_index}"
    logger.debug(f"Received response '{agent_chat_response}' ")
    return str(agent_chat_response)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

