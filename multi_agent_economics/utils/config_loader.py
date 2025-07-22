"""
Configuration loader utility for external data files.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Utility for loading configuration data from external files."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize with data directory path."""
        if data_dir is None:
            # Default to data directory relative to this file
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
    
    def get_tool_config_path(self) -> Optional[Path]:
        """Get path to tool configuration file."""
        path = self.data_dir / "config" / "enhanced_tools.json"
        return path if path.exists() else None
    
    def get_market_data_path(self) -> Optional[Path]:
        """Get path to market data file."""
        path = self.data_dir / "market_data" / "sector_growth_rates.json"
        return path if path.exists() else None
    
    def get_tool_params_path(self) -> Optional[Path]:
        """Get path to tool parameters file."""
        path = self.data_dir / "config" / "tool_parameters.json"
        return path if path.exists() else None
    
    def get_quality_config_path(self) -> Optional[Path]:
        """Get path to quality configuration file."""
        path = self.data_dir / "config" / "quality_thresholds.json"
        return path if path.exists() else None
    
    def get_prompt_templates_path(self) -> Optional[Path]:
        """Get path to prompt templates directory."""
        path = self.data_dir / "prompts"
        return path if path.exists() else None
    
    def get_role_definitions_path(self) -> Optional[Path]:
        """Get path to role definitions file."""
        path = self.data_dir / "prompts" / "role_definitions.json"
        return path if path.exists() else None
    
    def load_json_config(self, relative_path: str) -> Optional[Dict[str, Any]]:
        """Load a JSON configuration file."""
        path = self.data_dir / relative_path
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
        return None
    
    def load_text_config(self, relative_path: str) -> Optional[str]:
        """Load a text configuration file."""
        path = self.data_dir / relative_path
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
        return None


# Convenience function to create a loader with default data directory
def get_default_config_loader() -> ConfigLoader:
    """Get a config loader with the default data directory."""
    return ConfigLoader()
