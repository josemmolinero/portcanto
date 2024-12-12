"""
@ IOC - CEI_AB_M03_EAC6_2425S1 - JOSE MIGUEL GARCIA MOLINERO
"""

import sys
import logging
import shutil
import mlflow

import warnings

# Ignorar todos los warnings
warnings.filterwarnings("ignore")

from mlflow.tracking import MlflowClient
sys.path.append("..")
from clustersciclistes import (
    load_dataset,
    clean,
    extract_true_labels,
    clustering_kmeans,
    homogeneity_score,
    completeness_score,
    v_measure_score
)


def get_run_dir(artifacts_uri):
    """Return the path of the run."""
    return artifacts_uri[7:-10]


def remove_run_dir(run_dir):
    """Remove the directory using shutil."""
    shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(format="%(message)s", level=logging.INFO)

    # Create or get the experiment
    client = MlflowClient()
    experiment_name = "K sklearn ciclistes"
    exp = client.get_experiment_by_name(experiment_name)

    if not exp:
        mlflow.create_experiment(
            experiment_name,
            tags={"mlflow.note.content": "ciclistes variació de paràmetre K"}
        )
        mlflow.set_experiment_tag("version", "1.0")
        mlflow.set_experiment_tag("scikit-learn", "K")
        exp = client.get_experiment_by_name(experiment_name)

    mlflow.set_experiment(experiment_name)

    # Clean up existing runs
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    for run in runs:
        mlflow.delete_run(run.info.run_id)
        remove_run_dir(get_run_dir(run.info.artifact_uri))

    # Load dataset
    path_dataset = "./data/ciclistes.csv"
    ciclistes_data = load_dataset(path_dataset)

    # Clean dataset
    ciclistes_data_clean = clean(ciclistes_data)
    true_labels = extract_true_labels(ciclistes_data_clean)
    ciclistes_data_clean = ciclistes_data_clean.drop("tipus", axis=1)

    # Range of K values to test
    Ks = [2, 3, 4, 5, 6, 7, 8]

    # Iterate over different values of K
    for K in Ks:
        # Log dataset
        dataset = mlflow.data.from_pandas(ciclistes_data, source=path_dataset)

        # Start a new run for the current value of K
        mlflow.start_run(description=f"K={K}")
        mlflow.log_input(dataset, context="training")

        # Perform clustering
        clustering_model = clustering_kmeans(ciclistes_data_clean, K)
        data_labels = clustering_model.labels_

        # Compute metrics
        h_score = round(homogeneity_score(true_labels, data_labels), 5)
        c_score = round(completeness_score(true_labels, data_labels), 5)
        v_score = round(v_measure_score(true_labels, data_labels), 5)

        # Log metrics and parameters
        logging.info("K: %d", K)
        logging.info("H-measure: %.5f", h_score)
        logging.info("C-measure: %.5f", c_score)
        logging.info("V-measure: %.5f", v_score)

        tags = {
            "engineering": "JMGM-IOC",
            "release.candidate": "R1",
            "release.version": "1.0.0",
        }
        mlflow.set_tags(tags)

        # Log parameters and metrics
        mlflow.log_param("K", K)
        mlflow.log_metric("h", h_score)
        mlflow.log_metric("c", c_score)
        mlflow.log_metric("v_score", v_score)

        # Log artifact
        mlflow.log_artifact("./data/ciclistes.csv")

        # End the current run
        mlflow.end_run()

    print("s'han generat els runs")
