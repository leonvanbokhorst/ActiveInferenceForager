#!/bin/bash

# ActiveInferenceForager Project Setup Script

# Exit immediately if a command exits with a non-zero status
set -e

# Project name
PROJECT_NAME="ActiveInferenceForager"

# Create main project directory
mkdir $PROJECT_NAME
cd $PROJECT_NAME

# Create project structure
mkdir -p src tests docs examples config

# Create source code directory structure
mkdir -p src/$PROJECT_NAME/{agents,environments,utils}
touch src/$PROJECT_NAME/__init__.py
touch src/$PROJECT_NAME/agents/__init__.py
touch src/$PROJECT_NAME/environments/__init__.py
touch src/$PROJECT_NAME/utils/__init__.py

# Create main script
cat << EOF > src/$PROJECT_NAME/main.py
def main():
    print("Welcome to ActiveInferenceForager!")

if __name__ == "__main__":
    main()
EOF

# Create test directory structure
mkdir -p tests/{unit,integration}
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

# Create documentation files
touch docs/README.md
echo "# $PROJECT_NAME Documentation" > docs/README.md

# Create example scripts
touch examples/basic_simulation.py

# Create configuration files
touch config/default_config.yaml

# Create README file
cat << EOF > README.md
# $PROJECT_NAME

An adaptive AI agent implementing the Free Energy Principle and Active Inference in dynamic environments.

## Project Structure

\`\`\`
$PROJECT_NAME/
│
├── src/
│   └── $PROJECT_NAME/
│       ├── agents/
│       ├── environments/
│       └── utils/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
├── examples/
├── config/
│
├── README.md
├── requirements.txt
└── .gitignore
\`\`\`

## Setup

1. Clone the repository
2. Create a virtual environment: \`python -m venv venv\`
3. Activate the virtual environment:
   - On Windows: \`venv\Scripts\activate\`
   - On Unix or MacOS: \`source venv/bin/activate\`
4. Install the requirements: \`pip install -r requirements.txt\`

## Usage

[Provide basic usage instructions here]

## Contributing

[Provide contribution guidelines here]

## License

[Specify the license here]
EOF

# Create requirements file
touch requirements.txt

# Create .gitignore file
cat << EOF > .gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/

# IDE settings
.vscode/
.idea/
EOF

# Initialize git repository
git init

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install basic requirements
pip install numpy scipy matplotlib pyyaml pytest

# Freeze requirements
pip freeze > requirements.txt

echo "Project setup complete! Your $PROJECT_NAME project is ready to go."
echo "To get started, activate the virtual environment with 'source venv/bin/activate'"
