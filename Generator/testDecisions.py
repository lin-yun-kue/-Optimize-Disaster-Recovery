import time
from Generator.ProcessManager import ProcessManager
from Generator.ResultsParser import ResultsParser
from Generator.JSTRXGenerator import (
    JSTRXGenerator,
    QueueNode,
    CombiNode,
    ActivityCallbackData,
    AddToQueueAction,
    AssignAction,
    PostSimStateAction,
    Get,
)
from .expressions import Literal, Var


def handle_results(message: str):
    """Callback for RESULTS messages."""
    results = ResultsParser.parse(message)

    print(f"\n=== Simulation completed at time {results.sim_time} ===\n")

    print("Queue Nodes:")
    for name, stats in results.nodes.items():
        print(f"  {name}:")
        print(f"    Current: {stats.cur_count}, Total: {stats.tot_count}")
        print(f"    Max: {stats.max_cont}, Average: {stats.ave_cont:.2f}")

    print("\nActivity Nodes:")
    for name, stats in results.activities.items():
        print(f"  {name}:")
        print(f"    Instances: {stats.tot_inst} (current: {stats.cur_inst})")
        print(f"    Duration: {stats.ave_dur:.2f} (±{stats.sd_dur:.2f})")


get_call_index = 0

graph = JSTRXGenerator()
with graph:
    # create nodes
    crewWaiting = QueueNode(name="CrewWaiting", initialContent=10)

    putOutFireActivity = CombiNode(name="PutOutFire", duration=10)
    putOutFireSemaphore = QueueNode(name="PutOutFireSemaphore", initialContent=0)

    moveRubbleActivity = CombiNode(name="MoveRubble", duration=10)
    moveRubbleSemaphore = QueueNode(name="MoveRubbleSemaphore", initialContent=0)

    intermediateAct = CombiNode(name="IntermediateAct", duration=0)

    def decide_between_put_out_fire_and_move_rubble(data: ActivityCallbackData):
        print(f"[======>] Decide between PutOutFire and MoveRubble at t={data['sim_time']}")

    intermediateAct.onStart(decide_between_put_out_fire_and_move_rubble)
    intermediateAct.onStart(PostSimStateAction())

    # intermediateAct.onStart(AddToQueueAction(putOutFireSemaphore))

    def decider_get_action():
        global get_call_index
        path = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2][get_call_index]
        get_call_index = (get_call_index + 1) % 10
        print(f"Decider get action: {path}")
        # 1 for put out fire, 2 for move rubble
        return path

    graph.add_savevalue("tempVar", Literal(0))
    intermediateAct.onStart(AssignAction("tempVar", Get(decider_get_action)))
    graph.onIf(Var("tempVar").eq(1.0), AddToQueueAction(putOutFireSemaphore))
    graph.onIf(Var("tempVar").eq(2.0), AddToQueueAction(moveRubbleSemaphore))
    graph.onIf(Var("tempVar") > 0.0, AssignAction("tempVar", Literal(0.0)))

    crewWaiting.linkTo(intermediateAct)

    intermediateQueue = QueueNode(name="IntermediateQueue", initialContent=0)

    intermediateAct.linkTo(intermediateQueue)

    intermediateQueue.linkTo(putOutFireActivity)
    putOutFireSemaphore.linkTo(putOutFireActivity)

    intermediateQueue.linkTo(moveRubbleActivity)
    moveRubbleSemaphore.linkTo(moveRubbleActivity)

    finalFireCrew = QueueNode(name="FinalFireCrew", initialContent=0)
    finalRubbleCrew = QueueNode(name="FinalRubbleCrew", initialContent=0)

    putOutFireActivity.linkTo(finalFireCrew)
    moveRubbleActivity.linkTo(finalRubbleCrew)


full_path = graph.write_jstrx()

manager = ProcessManager()
manager.register_callback("MESSAGE", graph._post_callback)
manager.register_callback("GET", graph._get_callback)
manager.register_callback("RESULTS", handle_results)

now = time.time()

for i in range(100):
    manager.load_jstrx(full_path)
    manager.reset_model()
    manager.set_animate(False)

    manager.run_model(blocking=True)

print(f"100 simulations took {time.time() - now} seconds.")

manager.close()
manager.cleanup()


print("Simulation Finished")
