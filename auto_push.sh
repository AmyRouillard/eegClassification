#!/bin/bash

# Function to add, commit, and push changes
git_push() {
    git add .
    git commit -m "Auto-commit $(date +'%Y-%m-%d %H:%M:%S')"
    git push origin main  # Replace 'main' with your branch name if different
}

# Initial commit
git_push

# Loop for 7.5 hours (15 iterations with a 30-minute interval)
for ((i = 1; i <= 30; i++)); do
    sleep 1800  # Sleep for 30 minutes (1800 seconds)
    git_push
done
