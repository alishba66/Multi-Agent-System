import os
from dotenv import load_dotenv
from openai import AsyncOpenAI # type: ignore
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled # type: ignore

load_dotenv()
set_tracing_disabled(True)

provider = AsyncOpenAI(
    api_key= os.getenv("GEMINI_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",

)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash-exp",
    openai_client = provider,
)

web_dev = Agent(
    name = "web developer",
    instructions = "Built a responsive and performance websites using modern frameworks",
    model = model,
    handoff_description = "handoff to web developer if the task is related to web development."
)
mob_dev= Agent(
    name = "Mobile-developer",
    instructions = "Develop cross platform monile app for ios and android.",
    model = model,
    handoff_description = "handoff to mobile app developer if the task is related to mobile apps."
)

marketing_agent = Agent(
    name = "Marketing Agent",
    instructions = "create and execute marketing strategies for product launches.",
    model = model,
    handoff_description = "handoff to marketing agent if the task is related to marketing."
)

async def myAgent(user_input):
    manager = Agent(
        name = "Manager",
        instructions = "You will chat with the user and delegate tasks to speclized agents based on their requests.",
        model = model,
        handoff_description = [web_dev, mob_dev, marketing_agent, ]
    )

    response = await Runner.run(
        manager,
        input = user_input
    )

    return response.final_output
