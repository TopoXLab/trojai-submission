CUDA_VISIBLE_DEVICES=7 python entrypoint.py infer \
--model_filepath ./model/id-00000002/model.pt \
--result_filepath ./scratch/output.txt \
--scratch_dirpath ./scratch \
--examples_dirpath ./model/id-00000002/clean-example-data \
--round_training_dataset_dirpath /path/to/train-dataset \
--learned_parameters_dirpath ./learned_parameters \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--scale_parameters_filepath ./scale_params.npy


singularity build --fakeroot cyber-pdf-dec2022_sts_v2.simg trojan_detector.def


singularity run \
--bind /scr/lu/code/nist/trojai-cyber-pdf-main \
--nv \
./cyber-pdf-dec2022_sts_v2.simg \
infer \
--model_filepath=./model/id-00000002/model.pt \
--result_filepath=./output.txt \
--scratch_dirpath=./scratch/ \
--examples_dirpath=./model/id-00000002/clean-example-data/ \
--round_training_dataset_dirpath=/path/to/training/dataset/ \
--metaparameters_filepath=./metaparameters.json \
--schema_filepath=./metaparameters_schema.json \
--learned_parameters_dirpath=./learned_parameters/ \
--scale_parameters_filepath ./scale_params.npy



python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /scr/lu/code/nist/trojai-example/data/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /scr/lu/code/nist/trojai-example/data/cyber-pdf-dec2022-train/scale_params.npy


python entrypoint.py configure \
--scratch_dirpath ./scratch/ \
--metaparameters_filepath ./metaparameters.json \
--schema_filepath ./metaparameters_schema.json \
--learned_parameters_dirpath ./learned_parameters \
--configure_models_dirpath /scr/lu/code/nist/trojai-example/data/cyber-pdf-dec2022-train/models \
--scale_parameters_filepath /scr/lu/code/nist/trojai-example/data/cyber-pdf-dec2022-train/scale_params.npy \
--automatic_configuration


