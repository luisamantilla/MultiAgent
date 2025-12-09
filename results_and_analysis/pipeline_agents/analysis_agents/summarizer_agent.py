from openai import OpenAI
import json
import os
import pandas as pd
import pickle

from ...base_agent import BaseAgent


class SummarizerAgent(BaseAgent):
    """
    Reads a run_manifest.json, summarizes each output file (CSV, pickle, image), and saves a data_summary.json.
    Uses LLM to generate both a long summary for downstream agents and a concise summary (≤512 tokens) for memory.
    """

    def __init__(
        self,
        model="gpt-4.1",
        memory_manager=None,
        iteration: int = 0,
        goal: str = None,
        expertise: str = None,
    ):
        title = "Output Summarizer Agent"
        resolved_expertise = expertise or "Data Parsing, Context Extraction"
        resolved_goal = goal or "Summarize experiment outputs in a human-readable way."

        super().__init__(
            title=title,
            expertise=resolved_expertise,
            goal=resolved_goal,
            role="Data Summarizer and Reporter",  # Added a role
            model=model,
        )
        # self.title, self.expertise, self.goal are now set by BaseAgent
        # self.model is also set by BaseAgent
        self.client = OpenAI()  # OpenAI client might be redundant if BaseAgent initializes it, but keeping for now.
        self.memory_manager = memory_manager
        self.iteration = iteration

    def summarize_manifest(self, manifest_path, output_summary_path=None):
        with open(manifest_path) as mf:
            manifest = json.load(mf)
        # Compose a detailed summary of all files for the LLM
        file_summaries = []
        for file_info in manifest["files"]:
            path = file_info["path"]
            ftype = file_info["type"]
            desc = file_info["description"]
            if ftype == "csv":
                try:
                    df = pd.read_csv(path)
                    preview = df.head(3).to_dict()
                    stats = df.describe().to_dict()
                except Exception as e:
                    preview = {}
                    stats = {"error": str(e)}
                file_summaries.append(
                    {
                        "path": path,
                        "type": ftype,
                        "desc": desc,
                        "columns": list(df.columns) if "df" in locals() else [],
                        "preview": preview,
                        "stats": stats,
                    }
                )
            elif ftype == "pickle":
                try:
                    with open(path, "rb") as f:
                        obj = pickle.load(f)
                    keys = list(obj.keys()) if isinstance(obj, dict) else str(type(obj))
                except Exception as e:
                    keys = f"Error: {e}"
                file_summaries.append({"path": path, "type": ftype, "desc": desc, "keys": keys})
            elif ftype == "image":
                file_summaries.append({"path": path, "type": ftype, "desc": desc})
            else:
                file_summaries.append({"path": path, "type": ftype, "desc": desc})
        # LLM prompt for long summary
        context_prompt = (
            f"You are an expert data summarizer.\n"
            f"Given the following experiment output manifest and file previews, write a detailed, high-level summary of the experiment's results.\n"
            f"Focus on what was simulated, the main findings, and what files were created (with paths and types).\n"
            f"Manifest summary: {manifest.get('summary', '')}\n"
            f"File summaries: {json.dumps(file_summaries, indent=2)[:3000]}\n"
            f"Summarize in a way that is useful for a downstream analysis agent."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": context_prompt}],
            max_tokens=10000,
        )
        long_summary = response.choices[0].message.content
        # LLM prompt for short summary (≤512 tokens)
        summary_prompt = (
            f"Summarize the following experiment results in ≤512 tokens (about 2000 characters). "
            f"Focus on what was simulated, the main findings, and what files were created (with paths). "
            f"Be clear, self-contained, and concise.\n\n"
            f"---\n"
            f"{long_summary}\n"
            f"---\n"
            f"Summary (≤512 tokens):"
        )
        summary_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=512,
        )
        short_summary = summary_response.choices[0].message.content[:2000]
        # Save both summaries
        summary_obj = {
            "long_summary": long_summary,
            "short_summary": short_summary,
            "file_summaries": file_summaries,
        }
        if output_summary_path is None:
            output_summary_path = os.path.join(
                os.path.dirname(manifest_path), "data_summary.json"
            )
        with open(output_summary_path, "w") as sf:
            json.dump(summary_obj, sf, indent=2)
        # Save to memory if manager is provided
        if self.memory_manager:
            self.memory_manager.write_memory(
                iteration=self.iteration,
                agent="SummarizerAgent",
                summary=short_summary,
                output_file=output_summary_path,
            )
        return output_summary_path, summary_obj
