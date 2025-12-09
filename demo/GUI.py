# GUI.py
import gradio as gr
from orchestration import run_pipeline

def ui_run(question: str) -> str:
    """
    Gradio callback: run the full agent pipeline on `question`.
    """
    return run_pipeline(question)

if __name__ == "__main__":
    gr.Interface(
        fn=ui_run,
        inputs=gr.Textbox(
            label="Flu Hypothesis / Question",
            placeholder="e.g. Simulate the immune response to influenza infection…"
        ),
        outputs=gr.Textbox(label="Conversation & Summary", lines=25),
        title="Multi-Agent Flu Model Orchestrator",
        description="Enter a flu‐related question.",
        submit_btn="Submit",
        clear_btn="Clear"
    ).launch()
