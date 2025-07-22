"""
Quality tracking and production function system.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class QualityThreshold:
    """Defines quality thresholds for different products/services."""
    product_type: str
    high_quality_threshold: float
    medium_quality_threshold: float
    quality_tools: List[str]
    weights: Dict[str, float]  # Tool importance weights


class QualityFunction:
    """Implements quality production functions linking tool usage to output quality."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.thresholds: Dict[str, QualityThreshold] = {}
        
        # Load configuration from file if provided, otherwise use defaults
        if config_path and config_path.exists():
            self._load_from_config(config_path)
        else:
            self._setup_default_thresholds()
    
    def _load_from_config(self, config_path: Path):
        """Load quality thresholds from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for product_type, threshold_data in config.get("product_types", {}).items():
            threshold = QualityThreshold(
                product_type=product_type,
                high_quality_threshold=threshold_data["high_quality_threshold"],
                medium_quality_threshold=threshold_data["medium_quality_threshold"],
                quality_tools=threshold_data["quality_tools"],
                weights=threshold_data["weights"]
            )
            self.thresholds[product_type] = threshold
    
    def _setup_default_thresholds(self):
        """Setup default quality thresholds for financial products (fallback)."""
        
        # Structured Note quality thresholds
        structured_note = QualityThreshold(
            product_type="structured_note",
            high_quality_threshold=6.0,  # Credits spent on quality tools
            medium_quality_threshold=3.0,
            quality_tools=["sector_forecast", "price_note", "reflect"],
            weights={
                "sector_forecast": 0.5,  # Most important for quality
                "price_note": 0.4,
                "reflect": 0.1
            }
        )
        self.thresholds["structured_note"] = structured_note
        
        # Generic financial product
        generic_product = QualityThreshold(
            product_type="generic_financial",
            high_quality_threshold=4.0,
            medium_quality_threshold=2.0,
            quality_tools=["monte_carlo_var", "reflect"],
            weights={
                "monte_carlo_var": 0.8,
                "reflect": 0.2
            }
        )
        self.thresholds["generic_financial"] = generic_product
    
    def register_threshold(self, threshold: QualityThreshold):
        """Register a new quality threshold."""
        self.thresholds[threshold.product_type] = threshold
    
    def calculate_quality_score(self, product_type: str, tool_usage: Dict[str, float]) -> Dict[str, Any]:
        """Calculate quality score based on tool usage."""
        threshold = self.thresholds.get(product_type)
        if not threshold:
            return {"quality": "unknown", "score": 0.0, "reason": "No threshold defined"}
        
        # Calculate weighted quality score
        total_score = 0.0
        for tool, cost in tool_usage.items():
            if tool in threshold.quality_tools:
                weight = threshold.weights.get(tool, 0.0)
                total_score += cost * weight
        
        # Determine quality level
        if total_score >= threshold.high_quality_threshold:
            quality = "high"
        elif total_score >= threshold.medium_quality_threshold:
            quality = "medium"
        else:
            quality = "low"
        
        return {
            "quality": quality,
            "score": total_score,
            "threshold_high": threshold.high_quality_threshold,
            "threshold_medium": threshold.medium_quality_threshold,
            "tools_used": list(tool_usage.keys()),
            "quality_relevant_tools": threshold.quality_tools
        }
    
    def get_quality_requirements(self, product_type: str) -> Optional[Dict[str, Any]]:
        """Get quality requirements for a product type."""
        threshold = self.thresholds.get(product_type)
        if not threshold:
            return None
        
        return {
            "product_type": product_type,
            "high_quality_threshold": threshold.high_quality_threshold,
            "medium_quality_threshold": threshold.medium_quality_threshold,
            "required_tools": threshold.quality_tools,
            "tool_weights": threshold.weights
        }


class QualityTracker:
    """Tracks quality metrics across productions and agents."""
    
    def __init__(self, quality_function: QualityFunction):
        self.quality_function = quality_function
        self.production_records: List[Dict[str, Any]] = []
    
    def record_production(self, agent_id: str, product_type: str, 
                         tool_usage: Dict[str, float], external_actions: List[Dict[str, Any]]):
        """Record a production event with quality assessment."""
        quality_result = self.quality_function.calculate_quality_score(product_type, tool_usage)
        
        # Extract pricing information from external actions
        pricing_info = {}
        for action in external_actions:
            if action.get("action") == "post_price":
                pricing_info = {
                    "posted_price": action.get("price"),
                    "product": action.get("good"),
                    "timestamp": action.get("timestamp")
                }
                break
        
        record = {
            "agent_id": agent_id,
            "product_type": product_type,
            "quality_result": quality_result,
            "tool_usage": tool_usage,
            "pricing_info": pricing_info,
            "timestamp": datetime.now().isoformat(),
            "total_cost": sum(tool_usage.values())
        }
        
        self.production_records.append(record)
    
    def get_quality_distribution(self) -> Dict[str, Any]:
        """Get distribution of quality levels across all productions."""
        if not self.production_records:
            return {}
        
        quality_counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        total_productions = len(self.production_records)
        
        for record in self.production_records:
            quality = record["quality_result"]["quality"]
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
        
        return {
            "distribution": {
                quality: count / total_productions 
                for quality, count in quality_counts.items()
            },
            "counts": quality_counts,
            "total_productions": total_productions
        }
    
    def get_agent_quality_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get quality metrics for a specific agent."""
        agent_records = [r for r in self.production_records if r["agent_id"] == agent_id]
        
        if not agent_records:
            return {"agent_id": agent_id, "productions": 0}
        
        quality_scores = [r["quality_result"]["score"] for r in agent_records]
        quality_levels = [r["quality_result"]["quality"] for r in agent_records]
        total_costs = [r["total_cost"] for r in agent_records]
        
        return {
            "agent_id": agent_id,
            "productions": len(agent_records),
            "avg_quality_score": sum(quality_scores) / len(quality_scores),
            "quality_distribution": {
                level: quality_levels.count(level) / len(quality_levels)
                for level in ["high", "medium", "low"]
            },
            "avg_cost_per_production": sum(total_costs) / len(total_costs),
            "cost_efficiency": sum(quality_scores) / sum(total_costs) if sum(total_costs) > 0 else 0
        }
    
    def identify_quality_patterns(self) -> Dict[str, Any]:
        """Identify patterns in quality vs cost relationships."""
        if not self.production_records:
            return {}
        
        # Group by quality level
        quality_groups = {"high": [], "medium": [], "low": []}
        for record in self.production_records:
            quality = record["quality_result"]["quality"]
            if quality in quality_groups:
                quality_groups[quality].append(record)
        
        patterns = {}
        for quality, records in quality_groups.items():
            if not records:
                continue
            
            costs = [r["total_cost"] for r in records]
            scores = [r["quality_result"]["score"] for r in records]
            
            patterns[f"{quality}_quality"] = {
                "count": len(records),
                "avg_cost": sum(costs) / len(costs),
                "avg_score": sum(scores) / len(scores),
                "cost_range": [min(costs), max(costs)] if costs else [0, 0]
            }
        
        # Calculate correlation between cost and quality
        all_costs = [r["total_cost"] for r in self.production_records]
        all_scores = [r["quality_result"]["score"] for r in self.production_records]
        
        if len(all_costs) > 1:
            # Simple correlation calculation
            mean_cost = sum(all_costs) / len(all_costs)
            mean_score = sum(all_scores) / len(all_scores)
            
            numerator = sum((c - mean_cost) * (s - mean_score) for c, s in zip(all_costs, all_scores))
            denom_cost = sum((c - mean_cost) ** 2 for c in all_costs)
            denom_score = sum((s - mean_score) ** 2 for s in all_scores)
            
            correlation = numerator / (denom_cost * denom_score) ** 0.5 if denom_cost > 0 and denom_score > 0 else 0
            patterns["cost_quality_correlation"] = correlation
        
        return patterns
    
    def export_quality_report(self, output_file: str):
        """Export comprehensive quality analysis report."""
        report = {
            "summary": {
                "total_productions": len(self.production_records),
                "quality_distribution": self.get_quality_distribution(),
                "patterns": self.identify_quality_patterns()
            },
            "agent_metrics": {
                agent_id: self.get_agent_quality_metrics(agent_id)
                for agent_id in set(r["agent_id"] for r in self.production_records)
            },
            "detailed_records": self.production_records
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
