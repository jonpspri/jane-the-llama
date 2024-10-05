from config import ConfigurationSet, config_from_json
from os import path
from typing import Optional

from llama_deploy import (
    deploy_workflow,
    WorkflowServiceConfig,
    ControlPlaneConfig,
)

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from llama_index.core.response_synthesizers import (
        ResponseMode,
        get_response_synthesizer
        )

from llama_index.core.workflow import (
        Context,
        Workflow,
        Event,
        StartEvent,
        StopEvent,
        step
        )

from llama_index.embeddings.ibm import WatsonxEmbeddings
from llama_index.llms.ibm import WatsonxLLM
from llama_index.vector_stores.milvus import MilvusVectorStore

script_dir = path.dirname(path.abspath(__file__))
config = ConfigurationSet(
        config_from_json(path.join(script_dir, '../config.json'), read_from_file=True, ignore_missing_paths=True),
        config_from_json('/run/secrets/config', read_from_file=True, ignore_missing_paths=True),
        )

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RagWorkflow(Workflow):
    @step
    async def retrieve( self, ctx: Context, ev: StartEvent) -> Optional[RetrieverEvent]:
        """Entry point for RAG, triggered by a StartEvent with `query`."""

        query = ev.get("query")
        if not query:
            return None
        await ctx.set("query", query)

        vector_store = MilvusVectorStore(uri="http://standalone:19530",
                                 token=':'.join([
                                     config['milvus_username'],
                                     config['milvus_password']
                                     ]),
                                 dim=384, overwrite=False)
        index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=WatsonxEmbeddings(
                        model_id="ibm/slate-30m-english-rtrvr-v2",
                        url="https://us-south.ml.cloud.ibm.com",
                        apikey=config['watsonx_apikey'],
                        project_id=config['watsonx_project_id'],
                        truncate_input_tokens=512,
                        embed_batch_size=20
                    ))

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Return a response using the retrieved nodes."""
        llm = WatsonxLLM(
                model_id="ibm/granite-13b-chat-v2",
                url="https://us-south.ml.cloud.ibm.com",
                apikey=config['watsonx_apikey'],
                project_id="6a51b5ec-ff0d-4cca-9f25-7561888cd9cd",
                max_new_tokens=200
                )
        query = await ctx.get('query', default=None)
        synthesizer = get_response_synthesizer(
                llm=llm,
                response_mode=ResponseMode.COMPACT,
                use_async=True,
                streaming=False,
                )

        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

async def main():
    control_plane_config=ControlPlaneConfig()
    print(f"Control Plane URL: {control_plane_config.url}")
    await deploy_workflow(
        workflow=RagWorkflow(),
        workflow_config=WorkflowServiceConfig(
            service_name="jane_rag_workflow"
        ),
        control_plane_config=control_plane_config
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

