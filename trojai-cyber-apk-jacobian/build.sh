python entrypoint.py configure --automatic_configuration --scratch_dirpath=./scratch/ --metaparameters_filepath=./metaparameters.json --schema_filepath=./metaparameters_schema.json --learned_parameters_dirpath=./learned_parameters/ --configure_models_dirpath=/scr2/lu/TrojAI23/cyber-apk-nov2023-train/models


python entrypoint.py infer


singularity build --fakeroot cyber-apk-nov2023_sts_v0.simg detector.def


singularity build --fakeroot cyber-apk-nov2023_sts_v0.simg trojan_detector.def







