import json
from os import path
from time import sleep

from pymilvus import MilvusClient, MilvusException

script_dir = path.dirname(path.abspath(__file__))

with open(path.join(script_dir, '../secrets/milvus_root_password'), 'r') as file:
    milvus_root_password = file.read().rstrip()

with open(path.join(script_dir, '../secrets/jane_milvus_token'), 'r') as file:
    milvus_username, milvus_password = file.read().rstrip().split(':')

client = MilvusClient(
    # replace with your own Milvus server address
    uri='http://localhost:19530',
    token=f"root:{milvus_root_password}"
)

if not client.describe_user(user_name=milvus_username):
    client.create_user(
        user_name=milvus_username,
        password=milvus_password
    )

try:
    for privilege in client.describe_role(role_name="default_rw")['privileges']:
        client.revoke_privilege(
            role_name="default_rw",
            object_type=privilege['object_type'],
            object_name=privilege['object_name'],
            privilege=privilege['privilege']
            )
except MilvusException:
    client.create_role(role_name="default_rw")
    sleep(5)

privileges = json.load(open(path.join(script_dir, 'privileges.json'), 'r'))
for privilege in privileges:
    client.grant_privilege(
            role_name="default_rw",
            object_type=privilege['object_type'],
            object_name=privilege['object_name'],
            privilege=privilege['privilege']
            )

client.grant_role(
    user_name=milvus_username,
    role_name="default_rw"
    )
