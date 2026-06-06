from typing import Dict, Any, Protocol, Optional
import random

class DecisionAgent(Protocol):
    """
    DecisionAgent 是一個『規範』：
    只要有 decide(observation) 這個方法，
    就可以被 World 當成 decision agent 使用。
    """

    def decide(self, observation: Dict[str, Any]) -> Any:
        """
        observation:
            由 World 提供的狀態描述
        return:
            一個 action（由 World 解讀）
        """
        ...

class RandomDispatchAgent:
    def decide(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        trucks = list(observation["truck_locations"].keys())
        tasks = observation["pending_tasks"]

        if not trucks or not tasks:
            return None

        return {
            "type": "dispatch",
            "truck_id": random.choice(trucks),
            "task_id": random.choice(tasks),
        }
