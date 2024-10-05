import json
from os import path

from llama_index.llms.ibm import WatsonxLLM

script_dir = path.dirname(path.abspath(__file__))
config = json.load(open(path.join(script_dir, '../config.json'), 'r'))

llm = WatsonxLLM(
        model_id="ibm/granite-13b-chat-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=config['watsonx_apikey'],
        project_id=config['watsonx_project_id'],
        )
