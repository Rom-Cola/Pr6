import matplotlib
matplotlib.use('TkAgg')  # Використання TkAgg як бекенду для відображення графіків

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tkinter import Tk, Label, Entry, Button
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor

# Функція для підключення до MongoDB
def get_database():
    client = MongoClient("mongodb://localhost:27017/")
    return client["ant_colony_optimization"]

# Функція для збереження результатів у MongoDB
def save_to_mongodb(best_path, best_path_length):
    db = get_database()
    collection = db["optimization_results"]
    # Перетворення numpy.int64 у звичайний Python int
    best_path = [int(x) for x in best_path]
    best_path_length = float(best_path_length)
    result = {
        "best_path": best_path,
        "best_path_length": best_path_length
    }
    collection.insert_one(result)
    print("Results saved to MongoDB.")

def distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def ant_task(pheromone, points, alpha, beta):
    n_points = len(points)
    visited = [False] * n_points
    current_point = np.random.randint(n_points)
    visited[current_point] = True
    path = [current_point]
    path_length = 0

    while False in visited:
        unvisited = np.where(np.logical_not(visited))[0]
        probabilities = np.zeros(len(unvisited))

        for i, unvisited_point in enumerate(unvisited):
            probabilities[i] = pheromone[current_point, unvisited_point] ** alpha / distance(
                points[current_point], points[unvisited_point]) ** beta

        probabilities /= np.sum(probabilities)

        next_point = np.random.choice(unvisited, p=probabilities)
        path.append(next_point)
        path_length += distance(points[current_point], points[next_point])
        visited[next_point] = True
        current_point = next_point

    return path, path_length

def ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q, speed, visualize=True):
    n_points = len(points)
    pheromone = np.ones((n_points, n_points))
    best_path = None
    best_path_length = np.inf

    if visualize:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)
        ax.scatter(points[:, 0], points[:, 1], c='r', marker='o')

        for i, point in enumerate(points):
            ax.text(point[0], point[1], str(i), fontsize=12, ha='right')

        lines, = ax.plot([], [], c='g', linestyle='-', linewidth=2, marker='o')

        def init():
            lines.set_data([], [])
            return lines,

        def update(iteration):
            nonlocal best_path, best_path_length, pheromone

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(ant_task, pheromone, points, alpha, beta) for _ in range(n_ants)]
                results = [future.result() for future in futures]

            paths = [result[0] for result in results]
            path_lengths = [result[1] for result in results]

            for path, path_length in zip(paths, path_lengths):
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            pheromone *= evaporation_rate

            for path, path_length in zip(paths, path_lengths):
                for i in range(n_points - 1):
                    pheromone[path[i], path[i + 1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length

            x_data = [points[best_path[i], 0] for i in range(n_points)] + [points[best_path[0], 0]]
            y_data = [points[best_path[i], 1] for i in range(n_points)] + [points[best_path[0], 1]]

            lines.set_data(x_data, y_data)
            return lines,

        ani = FuncAnimation(fig, update, init_func=init, frames=n_iterations, interval=1000//speed, blit=True)
        plt.show()

    else:
        for iteration in range(n_iterations):
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(ant_task, pheromone, points, alpha, beta) for _ in range(n_ants)]
                results = [future.result() for future in futures]

            paths = [result[0] for result in results]
            path_lengths = [result[1] for result in results]

            for path, path_length in zip(paths, path_lengths):
                if path_length < best_path_length:
                    best_path = path
                    best_path_length = path_length

            pheromone *= evaporation_rate

            for path, path_length in zip(paths, path_lengths):
                for i in range(n_points - 1):
                    pheromone[path[i], path[i + 1]] += Q / path_length
                pheromone[path[-1], path[0]] += Q / path_length

    print("Best Path:", best_path, " path length:", best_path_length)
    save_to_mongodb(best_path, best_path_length)

    return best_path, best_path_length, pheromone  # Повертаємо найкращий шлях, його довжину та феромони

def start_optimization():
    try:
        n_ants = int(entry_n_ants.get())
        n_iterations = int(entry_n_iterations.get())
        alpha = float(entry_alpha.get())
        beta = float(entry_beta.get())
        evaporation_rate = float(entry_evaporation_rate.get())
        Q = float(entry_Q.get())
        n_points = int(entry_n_points.get())
        speed = int(entry_speed.get())

        points = np.random.rand(n_points, 2)  # Генерація випадкових 2D точок
        ant_colony_optimization(points, n_ants, n_iterations, alpha, beta, evaporation_rate, Q, speed)
    except ValueError:
        print("Please enter valid inputs.")

if __name__ == '__main__':
    root = Tk()
    root.title("Ant Colony Optimization")

    Label(root, text="Number of Ants:").grid(row=0, column=0)
    entry_n_ants = Entry(root)
    entry_n_ants.grid(row=0, column=1)
    entry_n_ants.insert(0, "10")

    Label(root, text="Number of Iterations:").grid(row=1, column=0)
    entry_n_iterations = Entry(root)
    entry_n_iterations.grid(row=1, column=1)
    entry_n_iterations.insert(0, "100")

    Label(root, text="Alpha:").grid(row=2, column=0)
    entry_alpha = Entry(root)
    entry_alpha.grid(row=2, column=1)
    entry_alpha.insert(0, "1.0")

    Label(root, text="Beta:").grid(row=3, column=0)
    entry_beta = Entry(root)
    entry_beta.grid(row=3, column=1)
    entry_beta.insert(0, "1.0")

    Label(root, text="Evaporation Rate:").grid(row=4, column=0)
    entry_evaporation_rate = Entry(root)
    entry_evaporation_rate.grid(row=4, column=1)
    entry_evaporation_rate.insert(0, "0.5")

    Label(root, text="Q:").grid(row=5, column=0)
    entry_Q = Entry(root)
    entry_Q.grid(row=5, column=1)
    entry_Q.insert(0, "1.0")

    Label(root, text="Number of Points:").grid(row=6, column=0)
    entry_n_points = Entry(root)
    entry_n_points.grid(row=6, column=1)
    entry_n_points.insert(0, "10")

    Label(root, text="Speed (iterations/sec):").grid(row=7, column=0)
    entry_speed = Entry(root)
    entry_speed.grid(row=7, column=1)
    entry_speed.insert(0, "1")

    Button(root, text="Start Optimization", command=start_optimization).grid(row=8, columnspan=2)

    root.mainloop()
