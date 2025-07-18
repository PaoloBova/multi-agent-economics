# Multi-Agent Economics Simulation

A sophisticated multi-agent system powered by Large Language Models (LLMs) for simulating economic interactions, market dynamics, and emergent behaviors in artificial economies.

## Overview

This project leverages Microsoft's AutoGen framework to create autonomous agents that interact in economic scenarios, enabling researchers and practitioners to study:

- Market dynamics and price formation
- Agent decision-making under uncertainty
- Emergent economic behaviors
- Policy impact simulation
- Resource allocation mechanisms

## Features

- **Multi-Agent Framework**: Built on AutoGen for robust agent communication
- **Economic Modeling**: Sophisticated economic models and market mechanisms
- **Data Versioning**: DVC integration for experiment tracking and reproducibility
- **LLM Integration**: Support for multiple LLM providers (OpenAI, etc.)
- **Analysis Tools**: Built-in tools for simulation analysis and visualization

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/paolobova/multi-agent-economics.git
   cd multi-agent-economics
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run a simple simulation**:
   ```bash
   poetry run python examples/basic_market.py
   ```

## Project Structure

```
multi-agent-economics/
├── multi_agent_economics/    # Main package
│   ├── agents/              # Agent implementations
│   ├── environments/        # Economic environments
│   ├── models/             # Economic models
│   └── utils/              # Utilities and helpers
├── examples/               # Example simulations
├── tests/                  # Test suite
├── docs/                   # Documentation
└── data/                   # Data and results
```

## Requirements

- Python 3.12+
- Poetry for dependency management
- OpenAI API key (or other LLM provider)

## Development

This project uses:
- **Poetry** for dependency management
- **Black** for code formatting
- **pytest** for testing
- **mypy** for type checking
- **DVC** for data versioning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `poetry run pytest`
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{multi_agent_economics,
  title={Multi-Agent Economics Simulation},
  author={Paolo Bova},
  year={2025},
  url={https://github.com/paolobova/multi-agent-economics}
}
```
