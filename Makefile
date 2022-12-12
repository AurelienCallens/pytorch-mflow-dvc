run_pipeline:
	export MLFLOW_RUN_ID=`python src/utils/start_pipeline.py $(RUN_NAME)`; \
	dvc repro