run-particle:
	@echo "Optimizing for a particle in 3D"
	python optimizing_particle.py --trace -do -dv -ss 500 -os 20 -dp

run-particle-penalty-based-collision:
	@echo "Optimizing for a particle with a penalty based collision model"
	python optimizing_particle_penalty_based.py --trace -do -dv -ss 500 -os 20 -dp

run-soft-2d:
	@echo "Optimizing for a soft body in 2D"
	python optimizing_soft_body_2d.py --sim-steps 300 -os 10 -dv -do

run-soft-3d:
	@echo "Optimizing for a soft body in 3D"
	python optimizing_soft_body_penalty_based.py --sim-steps 300 -os 25 -do -dv -dp

run-rigid:
	@echo "Optimizing for a rigid body in 3D"
	python optimizing_rigid_body.py --trace -ss 400 -os 10 -dv -dp -do
