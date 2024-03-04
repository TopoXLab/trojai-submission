python entrypoint.py configure --automatic_configuration --scratch_dirpath=./scratch/ --metaparameters_filepath=./metaparameters.json --schema_filepath=./metaparameters_schema.json --learned_parameters_dirpath=./learned_parameters/ --configure_models_dirpath=/scr2/lu/TrojAI23/cyber-apk-nov2023-train/models


python entrypoint.py infer


singularity build --fakeroot cyber-network-c2-feb2024_sts_v0.simg detector.def
