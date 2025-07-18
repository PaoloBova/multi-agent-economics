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

---

*This document should be updated as new best practices are discovered during development.*
