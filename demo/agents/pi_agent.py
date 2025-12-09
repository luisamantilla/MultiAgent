# agents/pi_agent.py
import os
from openai import OpenAI
from agents.agents import PI_AGENT

def call_pi_agent(question: str) -> str:
    """
    Send the user’s question + PI’s system prompt into the LLM
    and return the LLM’s final summary.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=PI_AGENT.model,
        messages=[
            PI_AGENT.system_message(),
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
