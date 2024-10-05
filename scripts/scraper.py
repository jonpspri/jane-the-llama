import json
import logging

from os import path

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.readers.wordpress import WordpressReader
from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.vector_stores.milvus import MilvusVectorStore

logger = logging.getLogger()
script_dir = path.dirname(path.abspath(__file__))
config = json.load(open(path.join(script_dir, '../config.json'), 'r'))

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

url = "https://www.springermusicstudio.com"
logger.info(f"Scraping {url} ...")
wp_reader = WordpressReader(url)
pages = wp_reader.load_data()
logger.info(f"{len(pages)} pages scraped")

vector_store = MilvusVectorStore(uri="http://localhost:19530",
                                 token=':'.join([
                                     config['milvus_username'],
                                     config['milvus_password']
                                     ]),
                                 dim=384, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(pages, storage_context=storage_context)
