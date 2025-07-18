# Development Best Practices & Setup Guide

This document captures the best practices and lessons learned during the setup of this multi-agent economics simulation project.

## Python Project Setup Best Practices

### 1. Environment Management
- **Use mamba/conda for Python version management**: `mamba create -n project-name` then `mamba install python=3.12`
- **Use Poetry for dependency management**: Poetry cannot install Python versions, only manage dependencies
- **Choose modern Python versions**: Python 3.12 recommended (July 2025) for performance and future-proofing
- **Avoid Python 3.13**: Too new, potential compatibility issues with some packages

### 2. Dependency Selection Strategy
- **Start minimal**: Only add dependencies you actually need
- **Essential core**: AutoGen, Pydantic, python-dotenv, pandas, numpy, matplotlib
- **Development tools**: pytest, black, isort, flake8, mypy, sphinx
- **Avoid bloat**: Skip seaborn, plotly, rich, loguru, etc. until actually needed
- **Use optional dependencies**: Define extras in `pyproject.toml` for optional features

### 3. pyproject.toml Configuration
```toml
# Keep consistent Python versions everywhere:
[tool.poetry.dependencies]
python = "^3.12"

[tool.black]
target-version = ['py312']

[tool.mypy]
python_version = "3.12"

# Classifiers should match
classifiers = [
    "Programming Language :: Python :: 3.12",
]
```

### 4. Code Quality Tools Configuration
- **Black + isort**: Automatic formatting with `profile = "black"` in isort
- **MyPy**: Strict type checking with comprehensive warnings enabled
- **Flake8**: Linting for code quality
- **Skip pre-commit initially**: Add later when workflow is established

### 5. Project Structure
```
project-name/
├── pyproject.toml          # Poetry config with minimal dependencies
├── README.md               # Required by Poetry
├── .env.example           # Environment variables template
├── .gitignore             # Python, IDE, OS ignores
├── docs/                  # Documentation
├── multi_agent_economics/ # Main package
│   ├── __init__.py
│   ├── agents/
│   ├── environments/
│   └── models/
├── tests/                 # Test suite
├── data/                  # Data files (DVC managed)
└── results/               # Output data
```

## Framework Selection Process

### Multi-Agent Framework Analysis
We evaluated multiple frameworks before choosing AutoGen:

1. **Microsoft AutoGen v0.4** ✅ **CHOSEN**
   - Mature, well-maintained by Microsoft
   - Excellent LLM integration and tool calling
   - Strong documentation and community
   - Built for production use

2. **CrewAI** ⚠️
   - Simpler but less mature
   - Good for basic use cases
   - Limited advanced features

3. **LangGraph** ⚠️
   - Powerful but complex
   - Steep learning curve
   - Better for graph-based workflows

4. **Custom Implementation** ❌
   - Too much reinventing the wheel
   - Maintenance burden

### Selection Criteria
- **Maturity & Maintenance**: Active development, enterprise backing
- **LLM Integration**: Native support for multiple providers
- **Documentation**: Comprehensive guides and examples
- **Community**: Active support and ecosystem
- **Production Ready**: Battle-tested in real applications

## Context Window Management for AI Agents

### Strategies for Efficient Conversations

#### 1. Conversation Reset Strategy
- **When to reset**: When context becomes summarized or responses get verbose
- **How to reset**: Start fresh conversation with current state summary
- **What to preserve**: Current file paths, completed tasks, next steps

#### 2. State-Based Development
- **Use workspace as memory**: Check files with `read_file` instead of assuming
- **Verify current state**: Use `list_dir` to see actual structure
- **Don't rely on conversation history**: Files are the source of truth

#### 3. Tool Call Optimization
- **Batch similar operations**: Create multiple files/directories at once
- **Be specific**: Target exact lines with `read_file` ranges
- **Minimize round trips**: Include all necessary context in single calls

#### 4. Focus Strategies
- **One task at a time**: Complete current step before planning next
- **Avoid over-explanation**: Skip verbose descriptions for obvious actions
- **Reference efficiently**: Use file attachments instead of quoting large code blocks

#### 5. Error Recovery
- **Quick fixes first**: Address immediate blockers (missing files, syntax errors)
- **Incremental progress**: Make small, testable changes
- **Validate frequently**: Test after each major change

### Context Reset Checklist
Before starting a new conversation, document:
- [ ] Current working directory
- [ ] Virtual environment status (mamba env name)
- [ ] Poetry installation status
- [ ] Key files that exist
- [ ] Last completed task
- [ ] Next planned task
- [ ] Any error states or blockers

## Error Patterns & Solutions

### Common Setup Issues
1. **Poetry can't find Python**: Install Python in mamba environment first
2. **Missing README.md**: Poetry expects this file to exist
3. **Version mismatches**: Keep all Python version references consistent
4. **Import errors**: Create `__init__.py` files in package directories

### Debug Commands
```bash
# Environment verification
python --version && which python
poetry env info
poetry install --dry-run

# Project status
ls -la
git status
dvc status
```

## Next Steps Template

When context window gets full, use this template:

```markdown
## Project Status Summary
- **Environment**: Python 3.12 in mamba env 'multi-agent-economics'
- **Dependencies**: Poetry installed, AutoGen working
- **Structure**: Basic package created with __init__.py files
- **Git**: [initialized/not initialized]
- **DVC**: [configured/not configured]
- **Current task**: [describe what you're working on]
- **Blockers**: [any current issues]

## Files Ready
- pyproject.toml (configured)
- .env.example (template ready)
- README.md (basic)
- multi_agent_economics/ (package structure)

## Next Priority
[What should be tackled next]
```

## Tools & Commands Reference

### Essential Commands
```bash
# Environment setup
mamba create -n project-name
mamba activate project-name
mamba install python=3.12

# Poetry workflow
poetry install
poetry add package-name
poetry remove package-name
poetry show

# Code quality
black .
isort .
flake8 .
mypy .

# Testing
pytest
pytest --cov=multi_agent_economics

# Git workflow
git init
git add .
git commit -m "Initial project setup"

# DVC workflow
dvc init
dvc add data/
dvc repro
```

## Data Version Control (DVC) Best Practices

### 1. DVC Setup Strategy
- **Initialize after Git**: Always run `git init` before `dvc init`
- **Commit DVC files**: Add `.dvc/`, `.dvcignore`, `dvc.yaml`, and `dvc.lock` to Git
- **Structure first**: Create data directories before defining pipeline
- **Start simple**: Begin with basic 3-stage pipeline (prepare → simulate → analyze)

### 2. Directory Structure for DVC
```
data/
├── raw/              # Original input data (tracked by DVC)
├── interim/          # Intermediate processing results
└── processed/        # Final data ready for ML/simulation

results/
├── analysis/         # Statistical summaries and reports
└── plots/           # Generated visualizations

scripts/
├── prepare_data.py   # Stage 1: Data preparation
├── run_simulation.py # Stage 2: Main processing/simulation
└── analyze_results.py# Stage 3: Analysis and visualization
```

### 3. Pipeline Design Principles
- **Single responsibility**: Each stage should have one clear purpose
- **Dependency tracking**: Explicitly declare all inputs and outputs
- **Parameter management**: Use YAML configs for reproducible experiments
- **Incremental outputs**: Generate intermediate results for debugging

### 4. DVC Configuration Best Practices

#### dvc.yaml Structure
```yaml
stages:
  stage_name:
    cmd: python scripts/script_name.py
    deps:                    # Input dependencies
      - data/input/
      - scripts/script_name.py
    params:                  # Configuration parameters
      - config.yaml:
          - section.parameter
    outs:                    # Output files/directories
      - data/output/
    plots:                   # Visualization outputs
      - results/plots/chart.png
```

#### Parameter Files
- **Use YAML**: More readable than JSON for configuration
- **Nested structure**: Organize parameters logically by category
- **Version control**: Keep parameter files in Git, not DVC
- **Documentation**: Comment parameter purposes and valid ranges

### 5. Common DVC Patterns

#### Data Pipeline Stages
1. **prepare_data**: Clean, validate, and format raw data
2. **run_simulation/train_model**: Execute main computation
3. **analyze_results**: Generate metrics, plots, and reports

#### Dependency Management
- **Scripts**: Always include the script file as a dependency
- **Data**: Track input data directories or specific files
- **Config**: Reference specific config sections, not entire files
- **Code**: Include relevant package directories for complex projects

#### Output Management
- **Intermediate data**: Save to `data/interim/` for debugging
- **Final results**: Save to `results/` or `data/processed/`
- **Visualizations**: Generate plots in `results/plots/`
- **Models**: Save trained models to dedicated directory

### 6. DVC Workflow Commands

#### Development Workflow
```bash
# Check pipeline status
dvc status                  # See what needs to be run
dvc dag                     # Visualize pipeline dependencies

# Run pipeline
dvc repro                   # Run only changed stages
dvc repro --force          # Force re-run all stages
dvc repro stage_name       # Run specific stage

# Experiment tracking
dvc params diff            # Compare parameter changes
dvc metrics show           # Display current metrics
dvc plots show             # Generate plot comparisons
```

#### Data Management
```bash
# Track large files
dvc add data/large_file.csv

# Remote storage (optional)
dvc remote add -d myremote s3://bucket/path
dvc push                   # Upload data to remote
dvc pull                   # Download data from remote

# Experiment branches
git checkout -b experiment-name
# modify parameters
dvc repro
git add dvc.lock params.yaml
git commit -m "Experiment: description"
```

### 7. .dvcignore Patterns
```
# Python artifacts
__pycache__/
*.py[cod]
.venv/

# IDE files
.vscode/
.idea/

# Temporary files
*.tmp
.temp/

# Large binary files (track separately)
*.pkl
*.joblib
*.h5
```

### 8. Error Prevention & Debugging

#### Common Issues
1. **Missing dependencies**: Include all input files and scripts in `deps`
2. **Circular dependencies**: Avoid stages that depend on their own outputs
3. **Path issues**: Use relative paths from project root
4. **Permission errors**: Ensure scripts are executable (`chmod +x`)

#### Debugging Commands
```bash
# Verbose output
dvc repro --verbose

# Check specific stage
dvc repro stage_name --verbose

# Force regenerate pipeline
rm dvc.lock && dvc repro

# Validate pipeline definition
dvc dag --verbose
```

### 9. Integration with Development Workflow

#### Git Integration
- **Commit strategy**: Commit `dvc.yaml` and parameter files to Git
- **Ignore outputs**: Add DVC-tracked files to `.gitignore`
- **Branch experiments**: Use Git branches for different experiments
- **Tag releases**: Tag important experiment results

#### Code Organization
- **Script location**: Keep DVC scripts in dedicated `scripts/` directory
- **Import structure**: Scripts should import from main package
- **Error handling**: Scripts should handle errors gracefully
- **Logging**: Add appropriate logging for debugging

### 10. Scaling and Advanced Usage

#### Performance Optimization
- **Parallel execution**: Use `dvc repro --jobs 4` for parallel stages
- **Caching**: Leverage DVC's automatic caching for unchanged inputs
- **Remote compute**: Configure DVC for cloud execution environments

#### Collaboration
- **Remote storage**: Set up shared S3/Azure/GCS for team data sharing
- **Pipeline sharing**: Version control pipeline definitions
- **Experiment tracking**: Use DVC experiments for systematic parameter sweeps

---

*This document should be updated as new best practices are discovered during development.*
