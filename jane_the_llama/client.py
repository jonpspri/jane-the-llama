from llama_deploy import LlamaDeployClient, ControlPlaneConfig

async def main():
    client = LlamaDeployClient(ControlPlaneConfig())

    session = client.createSession()
    result = await session.run(workflow="rag_workflow", query="What instruments does Ms Bee teach?")
    await result.print_response_stream()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

