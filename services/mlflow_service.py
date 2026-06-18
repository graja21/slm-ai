import mlflow

mlflow.set_experiment("Financial Document Extraction")


def log_financial_extraction(model, prompt_version, input_length, execution_time, json_valid):
    with mlflow.start_run():
        mlflow.log_param("model", model)
        mlflow.log_param("task", "financial_extraction")
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_metric("input_length", input_length)
        mlflow.log_metric("execution_time_seconds", execution_time)
        mlflow.log_metric("json_valid", int(json_valid))