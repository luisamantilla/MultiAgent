import re
import os

def parse_agent_responses(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    # Pattern: [Agent Name] (iteration N):
    pattern = re.compile(r"\[(.*?) Agent\] \(iteration (\d+)\):\n", re.MULTILINE)
    matches = list(pattern.finditer(text))

    results = []
    for i, match in enumerate(matches):
        agent = match.group(1)
        iteration = int(match.group(2))
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        response = text[start:end].strip()
        results.append({
            "agent": agent,
            "iteration": iteration,
            "response": response
        })

    return results

if __name__ == "__main__":
    responses = parse_agent_responses("output.txt")
    output_dir = "output/agent_responses"
    os.makedirs(output_dir, exist_ok=True)

    for entry in responses:
        filename = f"{entry['agent'].lower()}_iteration_{entry['iteration']}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(entry["response"])
        print(f"Saved: {filepath}")

    print("\nAll agent responses have been saved individually.")
     # Save all agent responses
    for entry in responses:
        filename = f"{entry['agent'].lower()}_iteration_{entry['iteration']}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(entry["response"])

