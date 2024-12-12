"""
@ IOC - CEI_AB_M03_EAC6_2425S1 - JOSE MIGUEL GARCIA MOLINERO
"""

import unittest
import os
import pickle

from generardataset import generar_dataset
from clustersciclistes import load_dataset, clean, extract_true_labels, clustering_kmeans, homogeneity_score, completeness_score, v_measure_score


class TestGenerarDataset(unittest.TestCase):
    """
    Test cases for the 'generar_dataset' function.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set global variables for tests.
        """
        global mu_p_be, mu_p_me, mu_b_bb, mu_b_mb, sigma, dicc

        mu_p_be = 3240  # Mean time for good climbers
        mu_p_me = 4268  # Mean time for bad climbers
        mu_b_bb = 1440  # Mean time for good descenders
        mu_b_mb = 2160  # Mean time for bad descenders
        sigma = 240  # Standard deviation

        dicc = [
            {"name": "BEBB", "mu_p": mu_p_be, "mu_b": mu_b_bb, "sigma": sigma},
            {"name": "BEMB", "mu_p": mu_p_be, "mu_b": mu_b_mb, "sigma": sigma},
            {"name": "MEBB", "mu_p": mu_p_me, "mu_b": mu_b_bb, "sigma": sigma},
            {"name": "MEMB", "mu_p": mu_p_me, "mu_b": mu_b_mb, "sigma": sigma}
        ]

    def test_longitud_dataset(self):
        """
        Test the length of the generated dataset.
        """
        arr = generar_dataset(200, 1, dicc[0])
        self.assertEqual(len(arr), 200)

    def test_valors_mitja_tp(self):
        """
        Test the average value of 'tp'.
        """
        arr = generar_dataset(100, 1, dicc[0])
        arr_tp = [row['tp'] for row in arr]  # Acceder a 'tp' por clave
        tp_mig = sum(arr_tp) / len(arr_tp)
        self.assertLess(tp_mig, 3400)

    def test_valors_mitja_tb(self):
        """
        Test the average value of 'tb'.
        """
        arr = generar_dataset(100, 1, dicc[1])
        arr_tb = [row['tb'] for row in arr]  # Acceder a 'tb' por clave
        tb_mig = sum(arr_tb) / len(arr_tb)
        self.assertGreater(tb_mig, 2000)

class TestClustersCiclistes(unittest.TestCase):
    """
    Test cases for the clustering process.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the data and clustering model for the tests.
        """
        global ciclistes_data_clean, data_labels, clustering_model, true_labels

        # Load and prepare dataset
        path_dataset = './data/ciclistes.csv'
        cls.ciclistes_data = load_dataset(path_dataset)
        ciclistes_data_clean = clean(cls.ciclistes_data)
        true_labels = extract_true_labels(ciclistes_data_clean)
        ciclistes_data_clean = ciclistes_data_clean.drop('tipus', axis=1)

        # Train clustering model
        clustering_model = clustering_kmeans(ciclistes_data_clean)
        data_labels = clustering_model.labels_

        # Save clustering model
        os.makedirs('./model', exist_ok=True)
        with open('model/clustering_model.pkl', 'wb') as f:
            pickle.dump(clustering_model, f)

    def test_check_column(self):
        """
        Check if a specific column exists in the cleaned dataset.
        """
        self.assertIn('tp', ciclistes_data_clean.columns)

    def test_data_labels_length(self):
        """
        Check that the length of the cluster labels matches the dataset size.
        """
        self.assertEqual(len(data_labels), len(ciclistes_data_clean))

    def test_model_saved(self):
        """
        Check that the clustering model file exists.
        """
        self.assertTrue(os.path.isfile('./model/clustering_model.pkl'))

if __name__ == '__main__':
    unittest.main()
