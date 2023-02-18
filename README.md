# A case study for inverse simulation of 3D objects

## Goal

Optimize for initial simulation parameters so that an object will ultimately achieve a target location, fulfilling the constraints imposed by the loss function that is being optimized.

## Pre-requisites

To install the required packages, please run:

```bash
pip install -r requirements.txt
```

## Running the code

From the provided Makefile, pick a type of an object and collision model to use in simulation. For example, to model a simple sphere with a penalty collision model, run:

```bash
make run-particle-penalty-based-collision
```

## Results

Feel free to change simulation parameters provided in `Makefile` to see how the results change. For more detailed description of the parameters, please refer to the `utils.py` file. For instance, in case of particle, modifying simulation horizon from 500 to 200 time steps will result in inability to fulfill the constraint on the number of collisions with the ground plane.
