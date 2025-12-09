
import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def simulation_evaluation_agent(simulation_data: dict, system_environment: str, rules: str):
    system_prompt = (
        "You are a Simulation Evaluation Agent in a multi-agent biomedical lab. "
        "Your role is to analyze structured simulation outputs of cell-based models. "
        "You must evaluate the simulation strictly based on the following biological rules:\n\n"
        f"{rules}\n\n"
        "Only point out abnormal trends, unexpected values, or possible modeling issues that violate the rules given. Do NOT make any assumptions. "
        "Do not summarize or explain what looks correct — only report problems. "
        "Cell death/birth or gain/loss events are not included in the data — assume that any increases or decreases in total cell count are expected and should not be flagged as anomalies."
    )

    question = (
        f"Analyze the following simulation output for environment: {system_environment}.\n\n"
        "Identify anomalies, spikes, inconsistencies, or biologically implausible trends **only if they violate the listed rules** and state the id of the rule. Do not make assumptions. \n"
        "Do not provide a summary or explanation of normal behavior. If no consistencies are present, only state that.\n\n"
        f"{simulation_data}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
    )
    
    return response.choices[0].message.content