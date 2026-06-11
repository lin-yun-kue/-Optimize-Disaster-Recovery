from Generator.ProcessManager import ProcessManager
from Generator.ResultsParser import ResultsParser
from Generator.JSTRXGenerator import (
    ActivityNode,
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
    ...

    # Free Resources Nodes
    freeTrucks = QueueNode(name="FreeTrucks", initialContent=5)
    freeExcavators = QueueNode(name="FreeExcavators", initialContent=3)

    def notify_trucks(data: ActivityCallbackData):
        """This function is called when a truck is available in the freeTrucks queue."""
        print(f"[======>] TruckNotifier at t={data['sim_time']}")

    def notify_excavators(data: ActivityCallbackData):
        """This function is called when an excavator is available in the freeExcavators queue."""
        print(f"[======>] ExcavatorNotifier at t={data['sim_time']}")

    graph.onIf("freeTrucks.initialContent", AddToQueueAction(freeExcavators))

    # Dirt dump location
    dumpedDirtCount = QueueNode(name="DumpedDirtCount", initialContent=0)
    dumpActivity = ActivityNode(name="DumpDirt", duration=10)
    dumpActivity.linkTo(freeTrucks)
    dumpActivity.linkTo(dumpedDirtCount, drawAmount=15)  # TODO: Truck capacity

    # Landslide
    def landslide(startingDirt: int, dumpSite: ActivityNode):
        print("Landslide!")

        # 1. Internal starting queues
        onsiteTrucks = QueueNode(name="OnsiteTrucks", initialContent=0)
        onsiteExcavators = QueueNode(name="OnsiteExcavators", initialContent=0)

        # 2. Dirt at landslide
        dirt = QueueNode(name="Dirt", initialContent=startingDirt)

        # 3. Clearing Landslide activity
        clearingLandslide = CombiNode(name="ClearingLandslide", duration=10)  # TODO: Loading duration
        onsiteTrucks.linkTo(clearingLandslide)
        onsiteExcavators.linkTo(clearingLandslide)
        dirt.linkTo(clearingLandslide, drawAmount=15)  # TODO: Truck capacity

        # 4. Return Excavator
        clearingLandslide.linkTo(onsiteExcavators)

        # 5. Trucks drive to dump site
        driveTime = ActivityNode(name="DriveToDumpSite", duration=10)  # TODO: Based on landslide distance to dump site
        clearingLandslide.linkTo(driveTime)
        driveTime.linkTo(dumpSite)

    landslide(10, dumpActivity)


full_path = graph.write_jstrx()

manager = ProcessManager()
manager.register_callback("MESSAGE", graph._post_callback)
manager.register_callback("GET", graph._get_callback)
manager.register_callback("RESULTS", handle_results)

manager.load_jstrx(full_path)
manager.reset_model()
manager.set_animate(True)

# manager.run_model(blocking=True)

manager.close()
manager.cleanup()
