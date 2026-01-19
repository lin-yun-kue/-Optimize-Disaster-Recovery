from dataclasses import dataclass, field

@dataclass
class Job:
    job_id: int
    location: str
    ready_time: float
    status: str = "waiting"  
    # waiting / in_transit / completed
