import json
import logging
import sys
import warnings

from os import path

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger()
script_dir = path.dirname(path.abspath(__file__))
config = json.load(open(path.join(script_dir, '../config.json'), 'r'))
warnings.filterwarnings("ignore", 'conflict with protected namespace "model_"')

Settings.chunk_size = 300
Settings.chunk_overlap = 50
Settings.embed_model = WatsonxEmbeddings(
        model_id="ibm/slate-30m-english-rtrvr-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=config['watsonx_apikey'],
        project_id=config['watsonx_project_id'],
        truncate_input_tokens=512,
        embed_batch_size=20
        )
Settings.llm = WatsonxLLM(
        model_id="ibm/granite-13b-chat-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=config['watsonx_apikey'],
        project_id="6a51b5ec-ff0d-4cca-9f25-7561888cd9cd",
        max_new_tokens=200
        )

vector_store = MilvusVectorStore(uri="http://localhost:19530",
                                 token=':'.join([
                                     config['milvus_username'],
                                     config['milvus_password']
                                     ]),
                                 dim=384, overwrite=False)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine(streaming=True)
query_s = ' '.join(sys.argv[1:])
print(query_s)
response = query_engine.query(query_s)
response.print_response_stream()
