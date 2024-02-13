python entrypoint.py configure --automatic_configuration --scratch_dirpath=./scratch/ --metaparameters_filepath=./metaparameters.json --schema_filepath=./metaparameters_schema.json --learned_parameters_dirpath=./learned_parameters/ --configure_models_dirpath=/scr/TrojAI23/rl-lavaworld-jul2023/training/models


python entrypoint.py infer


singularity build --fakeroot rl-lavaworld-jul2023_sts_v0.simg detector.def
