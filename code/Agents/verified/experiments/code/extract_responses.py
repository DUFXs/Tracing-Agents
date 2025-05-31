import json
import os
import re

# Load the evaluation results JSON file to initialize issue-to-difficulty mapping
evaluation_results_file = "evaluation_results.json"
issue_to_difficulty = {}

if os.path.exists(evaluation_results_file):
    with open(evaluation_results_file, "r") as file:
        data = json.load(file)

    # Create initial mapping of issues to their difficulty category
    for category, issues in data.items():
        for issue in issues.keys():
            issue_to_difficulty[issue] = {"difficulty": category, "agents": {}}

# Folder containing JSON files to process
input_folder = "individual_results"  # Change this to the actual folder path

# Process each JSON file in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
        
        # Extract relevant information
        instance_id = data.get("instance_id")  # Issue name
        agent = data.get("agent")  # Agent name
        response_json = data.get("response")  # This is a JSON string

        classification_pattern = re.compile(r'"classification"\s*:\s*"([^"]+)"')

        # Convert response string to dictionary if it's valid JSON
        classification = "Unknown"
        if response_json:
            try:
                # Decode the response JSON (since it's stored as a string)
                response_data = json.loads(response_json)
                classification = response_data.get("classification", "Unknown")
            except json.JSONDecodeError:
                # Attempt regex extraction if JSON parsing fails
                match = classification_pattern.search(response_json)
                if match:
                    classification = match.group(1)
                else:
                    classification = "Error in response"

        # Ensure the issue exists in the dictionary
        if instance_id not in issue_to_difficulty:
            issue_to_difficulty[instance_id] = {"difficulty": "Unknown", "agents": {}}

        # Store the agent's response classification
        issue_to_difficulty[instance_id]["agents"][agent] = classification

# Save the enhanced dictionary
output_filename = "enhanced_issue_to_difficulty.json"
with open(output_filename, "w") as output_file:
    json.dump(issue_to_difficulty, output_file, indent=4)

print(f"Enhanced issue-to-difficulty mapping saved to {output_filename}")
