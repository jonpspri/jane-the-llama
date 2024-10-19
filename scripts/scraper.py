import logging

from os import path

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.wordpress import WordpressReader
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger()
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

url = "https://www.springermusicstudio.com"
logger.info(f"Scraping {url} ...")
wp_reader = WordpressReader(url)
pages = wp_reader.load_data()
logger.info(f"{len(pages)} pages scraped")

vector_store = MilvusVectorStore(uri="http://localhost:19530",
                                 token=milvus_token,
                                 dim=384, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(pages, storage_context=storage_context)
