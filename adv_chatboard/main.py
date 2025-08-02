import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

@cl.on_chat_start
async def start():
    # Reference: https://ai.google.dev/gemini-api/docs/openai
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    cl.user_session.set("chat history", [])
    cl.user_session.set("config", config)

    agent = Agent(
        name="Gemini Agent",
        instructions="You are a helpful assistant that can answer questions and provide information based on the user's input.",
        model=model
    )

    cl.user_session.set("agent", agent)

    await cl.Message(content="Welcome! The Gemini Agent is ready to assist you. You can start asking questions.").send()


@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Processing your request...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get('chat history') or []

    # Ensure history is a list
    if not isinstance(history, list):
        history = []

    history.append({"role": "user", "content": message.content})

    try:
        print(f"Running agent with message: {message.content}History: {history}")

        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config
        )

        response_content = result.final_output
        msg.content = response_content
        await msg.update()

        cl.user_session.set("chat history", result.to_input_list())

        print(f"User: {message.content}")
        print(f"Gemini Agent: {response_content}")

    except Exception as e:
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")

        
        
    