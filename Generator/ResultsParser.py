from dataclasses import dataclass
from typing import Optional, Dict, Any
import json


@dataclass
class NodeStats:
    """Statistics for a queue node."""

    cur_count: int
    tot_count: int
    ave_wait: float
    ave_cont: float
    min_cont: int
    max_cont: int


@dataclass
class ActivityStats:
    """Statistics for an activity/combi node."""

    tot_inst: int
    cur_inst: int
    ave_dur: float
    sd_dur: float
    min_dur: float
    max_dur: float
    ave_int: Optional[float]
    sd_int: Optional[float]
    min_int: float
    max_int: float


@dataclass
class SimulationResults:
    """Container for all simulation results."""

    sim_time: float
    nodes: Dict[str, NodeStats]
    activities: Dict[str, ActivityStats]
    raw_data: Dict[str, Any]


class ResultsParser:
    """Parse RESULTS messages from the simulation."""

    @staticmethod
    def parse(message: str) -> SimulationResults:
        """Parse a RESULTS message into a SimulationResults object."""
        data = json.loads(message)

        nodes = {}
        activities = {}
        sim_time = data.get("SimTime", 0.0)

        # Group the flat dot-notation keys by node name
        node_data: Dict[str, Dict[str, Any]] = {}
        for key, value in data.items():
            if key == "SimTime":
                continue

            parts = key.split(".")
            node_name = parts[0]

            if node_name not in node_data:
                node_data[node_name] = {}

            remaining_key = ".".join(parts[1:]) if len(parts) > 1 else "root"
            node_data[node_name][remaining_key] = value

        # Parse each node's data
        for node_name, node_dict in node_data.items():
            if "CurCount" in node_dict:
                # Its a queue node
                nodes[node_name] = ResultsParser._parse_queue_node(node_dict)
            elif "TotInst" in node_dict:
                # Its an activity/combi node
                activities[node_name] = ResultsParser._parse_activity_node(node_dict)

        return SimulationResults(
            sim_time=sim_time, nodes=nodes, activities=activities, raw_data=data
        )

    @staticmethod
    def _parse_queue_node(data: Dict[str, Any]) -> NodeStats:
        """Parse queue node statistics."""
        return NodeStats(
            cur_count=data.get("CurCount", 0),
            tot_count=data.get("CurCount.TotCount", 0),
            ave_wait=data.get("CurCount.TotCount.AveWait", 0.0),
            ave_cont=data.get("CurCount.TotCount.AveWait.AveCont", 0.0),
            min_cont=data.get("CurCount.TotCount.AveWait.AveCont.MinCont", 0),
            max_cont=data.get("CurCount.TotCount.AveWait.AveCont.MinCont.MaxCont", 0),
        )

    @staticmethod
    def _parse_activity_node(data: Dict[str, Any]) -> ActivityStats:
        """Parse activity/combi node statistics."""
        return ActivityStats(
            tot_inst=data.get("TotInst", 0),
            cur_inst=data.get("TotInst.CurInst", 0),
            ave_dur=data.get("TotInst.CurInst.AveDur", 0.0),
            sd_dur=data.get("TotInst.CurInst.AveDur.SDDur", 0.0),
            min_dur=data.get("TotInst.CurInst.AveDur.SDDur.MinDur", 0.0),
            max_dur=data.get("TotInst.CurInst.AveDur.SDDur.MinDur.MaxDur", 0.0),
            ave_int=data.get("TotInst.CurInst.AveDur.SDDur.MinDur.MaxDur.AveInt"),
            sd_int=data.get("TotInst.CurInst.AveDur.SDDur.MinDur.MaxDur.AveInt.SDInt"),
            min_int=data.get(
                "TotInst.CurInst.AveDur.SDDur.MinDur.MaxDur.AveInt.SDInt.MinInt", 0.0
            ),
            max_int=data.get(
                "TotInst.CurInst.AveDur.SDDur.MinDur.MaxDur.AveInt.SDInt.MinInt.MaxInt",
                0.0,
            ),
        )
