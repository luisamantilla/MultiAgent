# Configuration settings for the crosstalk project

AGENT_SETTINGS = {
    "biological_expert": {
        "name": "Biological Expert",
        "expertise": "Biology",
        "response_time": 2  # seconds
    },
    "experimental_expert": {
        "name": "Experimental Expert",
        "expertise": "Experimental Methods",
        "response_time": 3  # seconds
    },
    "computational_expert": {
        "name": "Computational Expert",
        "expertise": "Computational Biology",
        "response_time": 2.5  # seconds
    },
    "pi": {
        "name": "Principal Investigator",
        "response_time": 1  # seconds
    }
}

CONVERSATION_SETTINGS = {
    "topic": "Normal Biological Problem",
    "max_turns": 10,
    "turn_time_limit": 5  # seconds
}

# Default output directory for saved runs
DEFAULT_OUTPUT_DIR = "/home/labuser/Desktop/lab_member_projects/Bobby_Ni/tumor-tcell-exp/20250902_crosstalk"