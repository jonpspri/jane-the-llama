import json
from os import path
from pymilvus import MilvusClient

script_dir = path.dirname(path.abspath(__file__))
config = json.load(open(path.join(script_dir, '../config.json'), 'r'))

client = MilvusClient(
    uri='http://localhost:19530',  # replace with your own Milvus server address
    token="root:Milvus"
)

client.update_password(
    user_name="root",
    old_password="Milvus",
    new_password=config['milvus_root_password']
)

