#!/bin/bash

# Shell script to run several python files and display a completion message in the end.

files = ("gen_db_req_json.py", "fast_db_embed.py", "db.py", "chroma_db.py")

# Execute each python file
for file in "${files[@]}"
do
    python3 "$file"
    if [ $? -ne 0 ]; then
        exit 1 # Exit if any script failts
    fi
done

# Completion message
echo "All scripts executed successfully"