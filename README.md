# ConStrobe Wrapper for use in Decision Inference

This branch contains code that wraps the ConStrobe simulation engine allowing you to create a simulation state in Python and communicate back and forth with handled events, callbacks, and returned values.

[ConstrobeEngine.py](Generator/ConstrobeEngine.py) is a attempt to recreate the SimPyTest simulation engine from the main branch. The simulation is not feature complete due to limitations with ConStrobe. It uses the same ScenarioConfig to setup a simulation Graph that spawns disasters and moves resources around.

It currently uses a basic 'first disaster' policy but could be set up to use the trained models. It just needs the Observation tensor to be constructed in the callback.

[ConStrobe](constrobe.com) is a DES simulation engine that can simulate processes involving resources and timed activities.
