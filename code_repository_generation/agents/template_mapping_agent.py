from agents.base_agent import BaseAgent
import json

class TemplateMappingAgent(BaseAgent):
    """
    Agent to map new files in a predicted dependency graph to the most similar old file in the original dependency graph using LLM-based mapping only.
    """
    def __init__(self, old_dep_graph, old_root=None, llm_model="gpt-4.1"):
        super().__init__(
            title="Template Mapping Agent",
            expertise="Python codebase refactoring and file similarity",
            goal="Map new files to the best old template files for adaptation",
            role="Template mapping assistant",
            model=llm_model
        )
        self.old_dep_graph = old_dep_graph
        self.old_files = list(old_dep_graph.keys())
        self.old_root = old_root

    def map_all(self, new_files):
        """
        Returns a dict: {new_file: best_old_file or None}
        Uses LLM to map new files to old files based on names and dependency lists.
        """
        def file_with_deps(files, dep_graph):
            lines = []
            for f in files:
                deps = dep_graph.get(f, [])
                lines.append(f"{f} (depends on: {', '.join(deps) if deps else 'none'})")
            return lines
        new_files_with_deps = file_with_deps(new_files, getattr(self, 'new_dep_graph', {}))
        old_files_with_deps = file_with_deps(self.old_files, self.old_dep_graph)
        prompt = (
            "You are a Python codebase refactoring assistant. "
            "Given a list of new files (with their dependencies) and a list of old files (with their dependencies) from a project, "
            "map each new file to the most appropriate old file to use as a template for code adaptation. "
            "Choose the best match based on filename similarity, dependency overlap, and likely functional similarity. "
            "If no old file is a reasonable match, use null. "
            "Return a JSON object mapping each new file to the best old file (or null if none is appropriate). "
            "Output ONLY the JSON object, no explanation or markdown.\n\n"
            "NEW FILES (with dependencies):\n" + "\n".join(new_files_with_deps) + "\n\n"
            "OLD FILES (with dependencies):\n" + "\n".join(old_files_with_deps)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            mapping = json.loads(response.choices[0].message.content)
        except Exception as e:
            print("Error parsing LLM template mapping:", e)
            mapping = {nf: None for nf in new_files}
        return mapping
