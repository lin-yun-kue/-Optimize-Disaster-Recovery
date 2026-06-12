# ConStrobe Wrapper for use in Decision Inference

This branch contains code that wraps the ConStrobe simulation engine allowing you to create a simulation state in Python and communicate back and forth with handled events, callbacks, and returned values.

[ConstrobeEngine.py](Generator/ConstrobeEngine.py) is a attempt to recreate the SimPyTest simulation engine from the main branch. The simulation is not feature complete due to limitations with ConStrobe. It uses the same ScenarioConfig to setup a simulation Graph that spawns disasters and moves resources around.

It currently uses a basic 'first disaster' policy but could be set up to use the trained models. It just needs the Observation tensor to be constructed in the callback.

[JSTRXGenerator.py](Generator/JSTRXGenerator.py) Provides a Python API for creating JSTRX files with node graphs, queues, activities, edges, and generated code to handle callbacks, events, and get messages. Examples are in the `testDecisions.py`, `testLandslides.py` and most notably used in the `ConstrobeEngine.py` with `testEngine.py`.

[ProcessManager.py](Generator/ProcessManager.py) Spawns and manages message communication with the running ConStrobe process. This includes callback handling for callbacks created by the JSTRXGenerator.

[testEngine.py](Generator/testEngine.py) Run using `python -m Generator.testEngine`. This file runs a small scale test using the ConstrobeEngine with a simple scenario. It currently logs success any time the process exits, which isn't always a success. Overall it is incomplete but does run, showing how to use the tools and generators here.

[ConStrobe](constrobe.com) is a DES simulation engine that can simulate processes involving resources and timed activities.
