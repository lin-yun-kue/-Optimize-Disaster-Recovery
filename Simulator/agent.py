from typing import Dict, Any, Protocol


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

class LongestQueueAgent:
    """
    一個最簡單的 rule-based agent：
    - 選第一台可用卡車
    - 去 queue 最長的地點
    """

    # todo update from location logic
    def decide(self, observation):
        trucks = observation.get("available_trucks", [])
        queues = observation.get("queue_lengths", {})

        if not trucks or not queues:
            return None  # 代表不做任何決策

        # 找 queue 最長的站點
        # todo: implement a beta version 
        target_location = max(queues, key=lambda k: queues[k])
        dispatch_target = ""

        return {
            "truck_id": trucks[0],
            "from_location": target_location,
            "dispatch_target": dispatch_target
        }
