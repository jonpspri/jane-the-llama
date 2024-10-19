import logging
import sys
import warnings

from os import path

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger()
warnings.filterwarnings("ignore", 'conflict with protected namespace "model_"')

script_dir = path.dirname(path.abspath(__file__))

with open(path.join(script_dir, '../secrets/jane_watsonx_apikey'), 'r') as file:
    watsonx_apikey = file.read().rstrip()
with open(path.join(script_dir, '../secrets/jane_watsonx_project_id'), 'r') as file:
    watsonx_project_id = file.read().rstrip()
with open(path.join(script_dir, '../secrets/jane_milvus_token'), 'r') as file:
    milvus_token = file.read().rstrip()

Settings.chunk_size = 300
Settings.chunk_overlap = 50
Settings.embed_model = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=watsonx_apikey,
        project_id=watsonx_project_id,
        truncate_input_tokens=512,
        embed_batch_size=20
        )
Settings.llm = WatsonxLLM(
        model_id="ibm/granite-13b-chat-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=watsonx_apikey,
        project_id=watsonx_project_id,
        max_new_tokens=200
        )

vector_store = MilvusVectorStore(uri="http://localhost:19530",
                                 token=milvus_token,
                                 dim=384, overwrite=False)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(streaming=True)
query_s = ' '.join(sys.argv[1:])
print(query_s)
response = query_engine.query(query_s)
response.print_response_stream()
