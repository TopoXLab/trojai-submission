python entrypoint.py configure
--automatic_configuration
--scratch_dirpath=./scratch/
--metaparameters_filepath=./metaparameters.json
--schema_filepath=./metaparameters_schema.json
--learned_parameters_dirpath=./new_learned_parameters/
--configure_models_dirpath=/scr/TrojAI23/object-detection-feb2023-train/models


python entrypoint.py infer
