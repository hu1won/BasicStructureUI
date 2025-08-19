import mlflow, os
def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    run = mlflow.start_run(run_name=cfg["exp_name"])
    mlflow.log_params({"engine": cfg["engine"], **cfg["train"]})
    return run
