move files to openfoam folder (one step up from extras folder) to execute
check if there is need to create some kind of output folder in openfoam folder - it might throw errors if not created

Added data_generator.py, train_surrogate_model.py, predict_surrogate_model.py and cavity_surface_plots.py ---- this is working code but only thing was that here we used initial pressure as a variable which does not make more sense since for incompressible flows, the flow does not depend upon absolute value of pressure

cavity template consistes of variation in lid velocity and initial pressure - in main, we have replaced initial pressure with viscosity