from jane_the_llama import RagWorkflow

async def main():
    w = RagWorkflow()
    result = await w.run(query="What instruments does Ms Bee teach?")
    await result.print_response_stream()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

