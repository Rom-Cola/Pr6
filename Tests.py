import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from main import distance, save_to_mongodb, ant_colony_optimization

class TestAntColonyOptimization(unittest.TestCase):

    def test_distance(self):
        point1 = np.array([0, 0])
        point2 = np.array([3, 4])
        self.assertEqual(distance(point1, point2), 5)

    @patch('main.MongoClient')
    def test_save_to_mongodb(self, MockMongoClient):
        mock_client = MockMongoClient.return_value
        mock_db = mock_client["ant_colony_optimization"]
        mock_collection = mock_db["optimization_results"]

        best_path = [0, 1, 2, 3]
        best_path_length = 10.0

        save_to_mongodb(best_path, best_path_length)

        # Перевірка, чи викликалася функція insert_one з правильними аргументами
        mock_collection.insert_one.assert_called_once_with({
            "best_path": best_path,
            "best_path_length": best_path_length
        })

    def test_ant_colony_optimization(self):
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])

        n_ants = 5
        n_iterations = 10
        alpha = 1.0
        beta = 2.0
        evaporation_rate = 0.5
        Q = 100
        speed = 1

        best_path, best_path_length, _ = ant_colony_optimization(points, n_ants, n_iterations, alpha, beta,
                                                                 evaporation_rate, Q, speed, visualize=False)

        self.assertIsNotNone(best_path)
        self.assertIsInstance(best_path_length, float)
        self.assertGreater(len(best_path), 0)
        self.assertGreater(best_path_length, 0)

    def test_probability_calculation(self):
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        n_ants = 5
        n_iterations = 10
        alpha = 1.0
        beta = 2.0
        evaporation_rate = 0.5
        Q = 100
        speed = 1

        # Ми створимо підмножину функції ant_colony_optimization для тестування ймовірностей
        n_points = len(points)
        pheromone = np.ones((n_points, n_points))
        current_point = 0
        unvisited = [1, 2, 3]

        probabilities = np.zeros(len(unvisited))

        for i, unvisited_point in enumerate(unvisited):
            probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                points[current_point], points[unvisited_point]) ** beta

        probabilities /= np.sum(probabilities)

        # Перевірка, що сума ймовірностей дорівнює 1
        self.assertAlmostEqual(np.sum(probabilities), 1.0)

    def test_pheromone_update(self):
        points = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ])
        n_ants = 5
        n_iterations = 1
        alpha = 1.0
        beta = 2.0
        evaporation_rate = 0.5
        Q = 100
        speed = 1

        n_points = len(points)
        pheromone = np.ones((n_points, n_points))

        best_path, best_path_length, updated_pheromone = ant_colony_optimization(points, n_ants, n_iterations, alpha, beta,
                                                                                 evaporation_rate, Q, speed, visualize=False)

        # Перевірка, що феромони були оновлені
        self.assertTrue(np.any(updated_pheromone != 1.0))

    def test_random_points_generation(self):
        n_points = 10
        points = np.random.rand(n_points, 2)

        # Перевірка, що було згенеровано правильну кількість точок
        self.assertEqual(points.shape, (n_points, 2))

        # Перевірка, що всі координати точок в межах [0, 1]
        self.assertTrue(np.all(points >= 0.0) and np.all(points <= 1.0))

if __name__ == '__main__':
    unittest.main()
