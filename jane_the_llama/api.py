import uvicorn

from asyncio import Lock, Task, create_task
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from redis import Redis
from typing import AsyncGenerator, Self
from uuid import uuid4
from weakref import WeakValueDictionary

from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import (
        StreamingAgentChatResponse,
        ChatMode,
        )
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
                model_id=settings.llm_model_id,
                url="https://us-south.ml.cloud.ibm.com",
                apikey=settings.watsonx_apikey,
                project_id=settings.watsonx_project_id,
                max_new_tokens=400,
                additional_params={ "stop_sequences": [ "\n{" ] }
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
    def get(cls, session_id: str):
        if session_id in cls._instances:
            return cls._instances[session_id]

        session_str = redis.get(session_id)
        if (session_str is None):
            return None

        assert isinstance(session_str, str)
        session = Session.model_validate_json(session_str)
        return SessionImpl(session)

    def __init__(self, session: Session):
        self.session = session
        self.__class__._instances[session.id] = self
        self.chat_engine = vector_index.as_chat_engine(
                    chat_mode=ChatMode.CONDENSE_PLUS_CONTEXT,
                    llm=llm,
                    )
        self.responses : dict[int, Task] = {}

    @property
    def id(self):
        return self.session.id

    @property
    def chat_history(self):
        return self.session.chat_history

    def commit(self):
        redis.setex(self.id, TIMEOUT, self.session.model_dump_json())

    async def wrapped_gen(self,
                          chat_message_i: int,
                          response: StreamingAgentChatResponse,
                          ) -> AsyncGenerator[str, None]:
        assert response.achat_stream is not None
        async for chat_response in response.achat_stream:
            assert chat_response.delta is not None
            assert isinstance(chat_response.delta, str)

            self.chat_history[chat_message_i].content = chat_response.message.content
            yield chat_response.delta
        self.commit()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Location", "Jane-Chat-Message-ID"],
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
    session_impl = SessionImpl.get(session_id)
    assert session_impl.chat_history is not None

    chat_message_id = str(uuid4())
    u.additional_kwargs["chat_message_id"] = chat_message_id
    response.headers['Jane-Chat-Message-ID'] = chat_message_id

    if u.role == MessageRole.USER:
        task_id = str(uuid4())
        a_chat_history = session_impl.chat_history.copy()
        SessionImpl._tasks[task_id] = create_task(
                session_impl.chat_engine.astream_chat(u.content, a_chat_history)
                )
        u.additional_kwargs["response_task_id"] = task_id

    message_index = len(session_impl.chat_history)
    message_uri=f"/sessions/{session_id}/chat_messages/{message_index}"
    session_impl.chat_history.append(u)
    redis.setex(session_impl.id, TIMEOUT, session_impl.session.model_dump_json())
    response.headers['Location']=message_uri
    logger.trace(f"Returning message {message_uri}")

    return u

@app.get("/sessions/{session_id}/chat_messages")
def read_chat_messages(session_id: str) -> list[ChatMessage]:
    session_impl = SessionImpl.get(session_id)
    if (not session_impl):
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session_impl.chat_history

@app.get("/sessions/{session_id}/chat_messages/{index}")
def read_chat_message(session_id: str, index: int) -> ChatMessage:
    session_impl = SessionImpl.get(session_id)

    # TODO Can I replace this with a try and index out of range?
    if index < 0 or index > len(session_impl.chat_history):
        raise HTTPException(status_code=404, detail=f"Index {index} not found")
    return session_impl.chat_history[index]

@app.get("/sessions/{session_id}/chat_messages/{index}/response")
async def read_chat_message_response(session_id: str, index: int) -> StreamingResponse:
    session_impl = SessionImpl.get(session_id)
    task_id = session_impl.chat_history[ index ].additional_kwargs[ "response_task_id" ]
    logger.debug(f"Waiting for a reply for task {task_id}")
    agent_chat_response = await SessionImpl._tasks[ task_id ]

    assert isinstance(agent_chat_response, StreamingAgentChatResponse)
    cm = ChatMessage( role=MessageRole.ASSISTANT, content=str(agent_chat_response) )
    cm.additional_kwargs['chat_message_id'] = task_id
    new_index = len(session_impl.chat_history)
    session_impl.chat_history.append(cm)
    session_impl.commit()

    headers = {}
    headers['Location'] = f"/sessions/{session_impl.id}/chat_messages/{new_index}"
    headers['Jane-Chat-Message-ID'] = task_id
    logger.debug("Streaming response...")

    return StreamingResponse(
            session_impl.wrapped_gen(new_index, agent_chat_response),
            headers = headers,
            )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000)

