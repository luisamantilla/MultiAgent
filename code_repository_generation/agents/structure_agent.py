from agents.base_agent import BaseAgent
import os
import ast
import json

class StructureAnalyzerAgent(BaseAgent):
    def __init__(self, memory_manager=None, llm_model="gpt-4.1", db_dir=None, collection_name="structure_analysis_memory"):
        super().__init__(
            title="Structure Analyzer Agent",
            expertise="Python project structure and dependency analysis",
            goal="Analyze project structure and build dependency graphs using LLMs",
            role="Project structure analyzer",
            model=llm_model
        )


    def analyze_folder(self, folder_path):
        filepaths = []
        for dirpath, _, filenames in os.walk(folder_path):
            for fname in filenames:
                if fname.endswith('.py'):
                    filepaths.append(os.path.join(dirpath, fname))
        self.module_map = self.build_module_map(filepaths, folder_path)
        self.file_deps = self.build_dependency_graph_llm(filepaths)
    
        
        return filepaths

    def build_module_map(self, filepaths, base_dir):
        module_map = {}
        base_dir = os.path.abspath(base_dir)
        base_pkg = os.path.basename(base_dir.rstrip(os.sep))
        for fp in filepaths:
            rel_path = os.path.relpath(fp, base_dir)
            parts = rel_path[:-3].split(os.sep)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            mod = ".".join([base_pkg] + parts)
            module_map[mod] = fp
        return module_map

    def build_dependency_graph_llm(self, filepaths):
        summaries = []
        for fp in filepaths:
            summary_content = self.summarize_file(fp)
            summaries.append(summary_content)
            # Store individual file summaries in memory

        prompt = (
            "You are an expert Python project architect. "
            "Given the following summaries of Python files in a project, "
            "infer the dependency graph as a JSON mapping from filename to a list of dependent filenames or module names. "
            "Include all dependencies and imports you find in the summaries, including both 'import ...' and 'from ... import ...' statements, as well as external and standard library modules. "
            "Output ONLY the JSON object, no explanation or markdown.\n\n"
            "Each summary includes the file name, classes, functions, and imports.\n\n"
            "FILE SUMMARIES:\n" + "\n".join(summaries)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        try:
            dep_graph = json.loads(response.choices[0].message.content)
        except Exception as e:
            print("Error parsing LLM dependency graph:", e)
            dep_graph = {fp: [] for fp in filepaths}
        return dep_graph

    def summarize_file(self, fpath):
        with open(fpath, 'r') as f:
            code = f.read()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return f"File: {fpath}\n<unparseable file>"
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports.append(f"{node.module}.{alias.name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
        summary = (
            f"File: {os.path.basename(fpath)}\n"
            f"Classes: {classes}\n"
            f"Functions: {functions}\n"
            f"Imports: {imports}\n"
        )
        return summary
