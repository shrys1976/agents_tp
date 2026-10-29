import asyncio

from agents import Agent, Runner

from single_agent_multiple_mcp import instructions, servers

research_agent = Agent(
    name ="Research Agent",
    instructions="""
    You are a research assistant.
    Your role is to find research sources.
    """,
)

thinking_agent = Agent(
    name="Thinking Agent",
    instructions ="""
    You are a research assistant.
    Your role is to plan the research.
    """,
)

filesystem_agent = Agent(
    name="Filsystem Agent",
    instructions ="""
    You are a research assistant.
    Your role is to write the research plan as text file.
    """,
)


async def main() -> None:
    async with (
        servers[0] as research_srv,
        servers[1] as thinking_srv,
        servers[2] as fs_srv,
    ):
        goal= """
    Produce a research plan to find the book: 'Mahabharata by Bibek Debroy'
    """

        print("Running...", goal)
        research_agent.mcp_servers = [research_srv]
        result = await Runner.run(research_agent, goal)
        thinking_agent.mcp_servers = [thinking_srv]
        result = await Runner.run(thinking_agent, result.final_output)
        filesystem_agent.mcp_servers = [fs_srv]
        result = await Runner.run(filesystem_agent, result.final_output)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
