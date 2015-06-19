# cannon_rhk
Model

train_v1.py: Natreniras Cannon-ov model za izbrane zvezde, pri tem izpustis 1. zvezdo na seznamu.
Poganjas z:
mpirun -np 8 python train_v1.py

label_v1.py: Na podlagi modela iz train_v1.py izracunas labele za 1. zvezdo iz seznama in jih primerjas s pravimi vrednostmi.

plot_results.py: Rises rezultate iz train_v1.py (theta vrednosti, primerjava med modelskimi in pravimi fluxi itd.).

