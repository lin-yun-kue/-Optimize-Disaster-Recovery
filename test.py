from ConstrobeManager import ConStrobeManager

sim = ConStrobeManager("TestCommunication.jstrx", verbose=True)

sim.set_attributes(Soil=15000, ExcWt=3, TrkWt=10)
sim.run_model(animate=True)
sim.write_get_results()
result = sim.read_result()

print("Simulation Time:", result["SimTime"])

sim.close()
sim.cleanup()
