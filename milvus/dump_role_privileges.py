import json
import re
from os import path

from pymilvus import MilvusClient, MilvusException

script_dir = path.dirname(path.abspath(__file__))
config = json.load(open(path.join(script_dir, '../config.json'), 'r'))

client = MilvusClient(
    # replace with your own Milvus server address
    uri='http://localhost:19530',
    token=f"root:{config['milvus_root_password']}"
)

print(json.dumps(client.describe_role(role_name="default_rw")['privileges']))
