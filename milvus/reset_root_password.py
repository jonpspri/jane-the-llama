from os import path
from pymilvus import MilvusClient

script_dir = path.dirname(path.abspath(__file__))

with open(path.join(script_dir, '../secrets/milvus_root_password'), 'r') as file:
    milvus_root_password = file.read().rstrip()

client = MilvusClient(
    uri='http://localhost:19530',  # replace with your own Milvus server address
    token="root:Milvus"
)

client.update_password(
    user_name="root",
    old_password="Milvus",
    new_password=milvus_root_password
)

