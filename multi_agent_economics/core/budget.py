"""
Budget and credit management system for tracking agent resource consumption.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class CreditTransaction:
    """Represents a single credit transaction."""
    agent_id: str
    amount: float
    transaction_type: str  # "debit", "credit", "transfer"
    description: str
    timestamp: datetime
    balance_after: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "amount": self.amount,
            "transaction_type": self.transaction_type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "balance_after": self.balance_after
        }


class BudgetManager:
    """Manages budgets and credit allocation for agents and organizations."""
    
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self.transactions: List[CreditTransaction] = []
        self.org_budgets: Dict[str, float] = {}
    
    def initialize_budget(self, agent_id: str, initial_budget: float):
        """Initialize budget for an agent or organization."""
        self.balances[agent_id] = initial_budget
        
        # Record initial credit
        transaction = CreditTransaction(
            agent_id=agent_id,
            amount=initial_budget,
            transaction_type="credit",
            description="Initial budget allocation",
            timestamp=datetime.now(),
            balance_after=initial_budget
        )
        self.transactions.append(transaction)
    
    def debit(self, agent_id: str, amount: float, description: str) -> bool:
        """Debit credits from an agent's budget."""
        if agent_id not in self.balances:
            return False
        
        if self.balances[agent_id] < amount:
            return False  # Insufficient funds
        
        self.balances[agent_id] -= amount
        
        transaction = CreditTransaction(
            agent_id=agent_id,
            amount=-amount,
            transaction_type="debit",
            description=description,
            timestamp=datetime.now(),
            balance_after=self.balances[agent_id]
        )
        self.transactions.append(transaction)
        
        return True
    
    def credit(self, agent_id: str, amount: float, description: str):
        """Credit credits to an agent's budget."""
        if agent_id not in self.balances:
            self.balances[agent_id] = 0
        
        self.balances[agent_id] += amount
        
        transaction = CreditTransaction(
            agent_id=agent_id,
            amount=amount,
            transaction_type="credit",
            description=description,
            timestamp=datetime.now(),
            balance_after=self.balances[agent_id]
        )
        self.transactions.append(transaction)
    
    def transfer(self, from_agent: str, to_agent: str, amount: float, description: str) -> bool:
        """Transfer credits between agents."""
        if not self.debit(from_agent, amount, f"Transfer to {to_agent}: {description}"):
            return False
        
        self.credit(to_agent, amount, f"Transfer from {from_agent}: {description}")
        return True
    
    def charge_credits(self, agent_id: str, amount: float, description: str = "Credit charge") -> bool:
        """
        Charge credits from an agent's budget.
        
        This is a convenience method that maps to debit() for compatibility
        with tool interfaces that expect charge_credits().
        
        Args:
            agent_id: Agent or category ID to charge
            amount: Amount to charge
            description: Description of the charge
            
        Returns:
            bool: True if successful, False if insufficient funds
        """
        return self.debit(agent_id, amount, description)
    
    def get_balance(self, agent_id: str) -> float:
        """Get current balance for an agent."""
        return self.balances.get(agent_id, 0.0)
    
    def get_transaction_history(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get transaction history for an agent or all agents."""
        if agent_id:
            transactions = [t for t in self.transactions if t.agent_id == agent_id]
        else:
            transactions = self.transactions
        
        return [t.to_dict() for t in transactions]
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get summary of all budgets and transactions."""
        total_allocated = sum(self.balances.values())
        total_spent = sum(
            -t.amount for t in self.transactions 
            if t.transaction_type == "debit"
        )
        
        return {
            "total_allocated": total_allocated,
            "total_spent": total_spent,
            "remaining": total_allocated,
            "agents": dict(self.balances),
            "transaction_count": len(self.transactions)
        }


class CreditTracker:
    """Tracks credit usage patterns and efficiency metrics."""
    
    def __init__(self, budget_manager: BudgetManager):
        self.budget_manager = budget_manager
    
    def calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate credit efficiency metrics across agents."""
        metrics = {}
        
        for agent_id in self.budget_manager.balances:
            transactions = self.budget_manager.get_transaction_history(agent_id)
            debits = [t for t in transactions if t["transaction_type"] == "debit"]
            
            if not debits:
                continue
            
            total_spent = sum(-t["amount"] for t in debits)
            spending_frequency = len(debits)
            avg_transaction = total_spent / spending_frequency if spending_frequency > 0 else 0
            
            # Calculate spending velocity (credits per time unit)
            if len(debits) > 1:
                first_spend = datetime.fromisoformat(debits[0]["timestamp"])
                last_spend = datetime.fromisoformat(debits[-1]["timestamp"])
                time_span = (last_spend - first_spend).total_seconds() / 3600  # hours
                velocity = total_spent / max(time_span, 0.1)
            else:
                velocity = 0
            
            metrics[agent_id] = {
                "total_spent": total_spent,
                "spending_frequency": spending_frequency,
                "avg_transaction_size": avg_transaction,
                "spending_velocity": velocity,
                "remaining_budget": self.budget_manager.get_balance(agent_id)
            }
        
        return metrics
    
    def identify_spending_patterns(self) -> Dict[str, Any]:
        """Identify spending patterns and anomalies."""
        patterns = {}
        
        # Analyze tool usage patterns
        tool_spending = {}
        for transaction in self.budget_manager.transactions:
            if "tool:" in transaction.description:
                tool_name = transaction.description.split("tool:")[1].split()[0]
                if tool_name not in tool_spending:
                    tool_spending[tool_name] = []
                tool_spending[tool_name].append(-transaction.amount)
        
        patterns["tool_usage"] = {
            tool: {
                "total_spent": sum(amounts),
                "usage_count": len(amounts),
                "avg_cost": sum(amounts) / len(amounts)
            }
            for tool, amounts in tool_spending.items()
        }
        
        # Identify high/low spenders
        agent_totals = {}
        for agent_id in self.budget_manager.balances:
            transactions = self.budget_manager.get_transaction_history(agent_id)
            debits = [t for t in transactions if t["transaction_type"] == "debit"]
            agent_totals[agent_id] = sum(-t["amount"] for t in debits)
        
        if agent_totals:
            avg_spending = sum(agent_totals.values()) / len(agent_totals)
            patterns["spending_distribution"] = {
                "average": avg_spending,
                "high_spenders": [
                    agent for agent, total in agent_totals.items() 
                    if total > avg_spending * 1.5
                ],
                "low_spenders": [
                    agent for agent, total in agent_totals.items() 
                    if total < avg_spending * 0.5
                ]
            }
        
        return patterns
