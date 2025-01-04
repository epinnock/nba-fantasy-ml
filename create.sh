#!/bin/sh

# Create directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/optimization
mkdir -p tests

# Create empty Python files
touch src/__init__.py
touch src/data/__init__.py
touch src/data/collector.py
touch src/data/processor.py
touch src/features/__init__.py
touch src/features/builder.py
touch src/models/__init__.py
touch src/models/trainer.py
touch src/models/predictor.py
touch src/optimization/__init__.py
touch src/optimization/lineup.py

# Create configuration and requirements files
touch config.yaml
touch requirements.txt

echo "Project structure created successfully!"

# Print the directory structure
if command -v tree > /dev/null 2>&1; then
    tree .
else
    find . -type d | sed -e "s/[^-][^\/]*\//  |/g" -e "s/|\([^ ]\)/|-\1/"
fi