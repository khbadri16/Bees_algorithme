from __future__ import annotations
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def generate_equilateral_triangle(self, a: float) -> Triangle:
        p1 = Point(self.x, self.y + np.sqrt(3) * a / 3)
        p2 = Point(self.x + a / 2, self.y - np.sqrt(3) * a / 6)
        p3 = Point(self.x - a / 2, self.y - np.sqrt(3) * a / 6)
        return Triangle(p1, p2, p3)

    def __repr__(self):
        return f"{self.x},{self.y}"

    def draw(self, ax):
        ax.scatter([self.x], [self.y], color='r', s=10)


class Triangle:
    def __init__(self, p1: Point, p2: Point, p3: Point):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def area(self):
        return np.abs(
            self.p1.x * (self.p2.y - self.p3.y) +
            self.p2.x * (self.p3.y - self.p1.y) +
            self.p3.x * (self.p1.y - self.p2.y)
        ) / 2

    def draw(self, ax):
        coords = np.array([[self.p1.x, self.p1.y], [self.p2.x, self.p2.y], [self.p3.x, self.p3.y]])
        poly = Polygon(coords, alpha=0.3, edgecolor="blue", linewidth=1, facecolor="lightblue")
        ax.add_patch(poly)

    def contains(self, p: Point):
        t1 = Triangle(self.p1, self.p2, p)
        t2 = Triangle(self.p1, self.p3, p)
        t3 = Triangle(self.p2, self.p3, p)
        a = self.area()
        # Use a small tolerance for floating point errors
        return np.abs(a - (t1.area() + t2.area() + t3.area())) < 1e-6


class Covering:
    # --- Class-level (static) variables ---
    pts = None
    points = None
    triangles = None
    side = 15
    total_points_to_cover = 0
    UNCOVERED_PENALTY = 1000

    # --- NEW: The Coverage Lookup Map ---
    # This will store which antennas cover which points
    # point_coverage_map[i] = [list of antenna_ids that cover point i]
    point_coverage_map = []

    @staticmethod
    def prepare_data():
        """Initializes the static data for the problem."""
        print("Pre-calculating coverage map... (this happens once)")
        Covering.points = [Point(x, y) for x, y in Covering.pts]
        Covering.total_points_to_cover = len(Covering.points)
        Covering.triangles = [p.generate_equilateral_triangle(Covering.side) for p in Covering.points]

        # --- Build the lookup map ---
        Covering.point_coverage_map = []
        for i in range(Covering.total_points_to_cover):  # For each point i
            pt = Covering.points[i]
            antennas_that_cover_this_point = []
            for j in range(len(Covering.triangles)):  # For each antenna j
                if Covering.triangles[j].contains(pt):
                    antennas_that_cover_this_point.append(j)
            Covering.point_coverage_map.append(antennas_that_cover_this_point)
        print("Coverage map built.")

    def __init__(self, init=True):
        """
        A 'Food Source' (Solution) for the Bee Algorithm.
        The solution is a binary numpy array.
        1 = build an antenna at this point.
        0 = do not build.
        """
        if init:
            # A "Scout Bee" creates a random solution
            self.solution = np.random.randint(low=0, high=2, size=len(Covering.pts))
        self.fitness = float('inf')
        self.nb = 0  # Number of points covered
        self.sm = 0  # Number of antennas used

    def draw(self, title="Solution"):
        """Draws the final solution."""
        _, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(title)

        covered_points = set()

        # Draw the antennas (triangles) that are "on"
        for i in range(len(self.solution)):
            if self.solution[i] == 1:
                Covering.triangles[i].draw(ax)

        # Draw all the points
        for i in range(len(Covering.points)):
            p = Covering.points[i]
            is_covered = False
            # --- Use the fast map to check for coverage ---
            for antenna_id in Covering.point_coverage_map[i]:
                if self.solution[antenna_id] == 1:
                    is_covered = True
                    break

            # Draw covered points in green, uncovered in red
            color = 'g' if is_covered else 'r'
            marker = 'o' if is_covered else 'X'
            ax.scatter([p.x], [p.y], color=color, s=25, marker=marker, zorder=5)

        plt.ylim(0, 100)
        plt.xlim(0, 100)
        plt.show()

    def compute_fitness(self):
        """
        --- NEW: OPTIMIZED FITNESS FUNCTION ---
        This version uses the pre-calculated coverage map and is *extremely* fast.
        It no longer does any t.contains() or triangle math.
        """
        num_antennas = self.solution.sum()
        points_covered_count = 0

        # Loop through all points (e.g., 50 points)
        for i in range(Covering.total_points_to_cover):
            is_covered = False
            # Get the *short list* of antennas that can cover this point
            antennas_for_this_point = Covering.point_coverage_map[i]

            # Check if any of those antennas are "on" in our solution
            for antenna_id in antennas_for_this_point:
                if self.solution[antenna_id] == 1:
                    is_covered = True
                    break  # This point is covered, move to the next point

            if is_covered:
                points_covered_count += 1

        num_uncovered = Covering.total_points_to_cover - points_covered_count

        # Store for reporting
        self.nb = points_covered_count
        self.sm = num_antennas

        # The total cost is the number of antennas + a huge penalty for each failure
        self.fitness = num_antennas + (num_uncovered * Covering.UNCOVERED_PENALTY)

        return self.fitness

    def search_neighborhood(self, flips=2):
        """
        This is the "Forager Bee" local search operator.
        It creates a new solution in the "neighborhood" of this one
        by flipping a few random bits.
        """
        neighbor = Covering(init=False)  # Create an empty solution
        neighbor.solution = np.array(self.solution)  # Copy the solution

        # Flip 'flips' number of bits
        for _ in range(flips):
            idx = np.random.randint(0, len(self.solution))
            neighbor.solution[idx] = 1 - neighbor.solution[idx]  # Flip 0 to 1 or 1 to 0

        return neighbor

    def __str__(self):
        return f"Fitness: {self.fitness} (Antennas: {self.sm}, Covered: {self.nb}/{Covering.total_points_to_cover})"

    def __lt__(self, other):
        """Less-than, for sorting. Lower fitness is better."""
        return self.fitness < other.fitness


class BeeAlgorithm:
    def __init__(self, n, e, m, nep, nsp, max_iter):
        self.n = n  # Total population (food sources)
        self.e = e  # Elite sites
        self.m = m  # Best sites (non-elite)
        self.nep = nep  # Foragers for elite sites
        self.nsp = nsp  # Foragers for best sites
        self.max_iter = max_iter

        self.population = [Covering() for _ in range(self.n)]
        self.global_best = Covering()
        self.global_best.fitness = float('inf')

    def solve(self):
        print("--- Starting Bee Algorithm for Antenna Coverage ---")
        print(f"Parameters: n={self.n}, e={self.e}, m={self.m}, nep={self.nep}, nsp={self.nsp}")

        for i in range(self.max_iter):
            # 1. Evaluate all solutions in the population
            for bee in self.population:
                bee.compute_fitness()

            # 2. Sort population (lower fitness is better)
            self.population.sort()

            # 3. Update global best
            if self.population[0] < self.global_best:
                self.global_best = self.population[0]

            # Report progress
            if i % 50 == 0:
                print(f"Iter {i:>4}: {self.global_best}")

            # Check for perfect solution
            # --- MODIFICATION: As requested, commenting out the "perfect solution" stopping condition.
            # The algorithm will now *only* stop when max_iter is reached.
            # if self.global_best.nb == Covering.total_points_to_cover:
            #     print(f"\n--- PERFECT SOLUTION FOUND at Iteration {i}! ---")
            #     print(f"Best Solution: {self.global_best}")
            #     return self.global_best
            # --- END MODIFICATION ---

            # --- 4. Local Search (Exploitation) ---
            new_population = []

            # For 'e' elite sites, send 'nep' foragers
            for j in range(self.e):
                parent_bee = self.population[j]
                best_neighbor = parent_bee

                for k in range(self.nep):
                    neighbor = parent_bee.search_neighborhood()
                    neighbor.compute_fitness()
                    if neighbor < best_neighbor:
                        best_neighbor = neighbor

                new_population.append(best_neighbor)

            # For 'm' best sites, send 'nsp' foragers
            for j in range(self.m):
                parent_bee = self.population[self.e + j]
                best_neighbor = parent_bee

                for k in range(self.nsp):
                    neighbor = parent_bee.search_neighborhood()
                    neighbor.compute_fitness()
                    if neighbor < best_neighbor:
                        best_neighbor = neighbor

                new_population.append(best_neighbor)

            # --- 5. Global Search (Exploration) ---
            num_scouts = self.n - self.e - self.m
            for j in range(num_scouts):
                # Abandon worst sites and create new random scouts
                new_population.append(Covering())

                # 6. Replace old population
            self.population = new_population

        print(f"\n--- Max Iterations ({self.max_iter}) Reached ---")
        print(f"Best Solution Found: {self.global_best}")
        return self.global_best


# --- Main execution ---
if __name__ == "__main__":
    pts = [
        [0.21785146, 32.80884325], [36.5398575, 18.85454466], [30.22558699, 40.47690186],
        [18.24973109, 92.01706988], [58.08376113, 92.00152543], [46.93621006, 80.97613966],
        [47.32211679, 91.18953057], [40.65971946, 14.94515933], [97.7445341, 62.9737585, ],
        [67.15084079, 32.35608877], [68.75175516, 54.20477561], [91.80634184, 94.35959023],
        [45.40857899, 61.2605084, ], [99.48086417, 52.49122315], [48.93285306, 62.77433716],
        [26.90344815, 72.73694627], [60.28518974, 34.06750296], [21.23856624, 37.76163464],
        [85.79582818, 41.13272287], [81.11078688, 62.25071194], [83.08482713, 84.81411278],
        [35.2874055, 29.80941103], [49.66321866, 35.67015919], [70.37347844, 76.05067313],
        [91.9348889, 81.85162595], [73.46612228, 40.7045268, ], [38.3556454, 84.28602823],
        [5.21712579, 18.30743315], [75.72162521, 54.12073958], [26.72551212, 68.47890848],
        [81.41504844, 97.70267505], [19.59272594, 53.29613031], [82.88070594, 79.33406044],
        [42.97770388, 39.39407873], [84.12304215, 32.42914694], [88.1637349, 95.53768435],
        [58.76495944, 25.67892065], [62.05423696, 83.28110411], [60.85664933, 4.40774533],
        [19.72387992, 12.50406704], [85.68495556, 64.60429362], [45.40271382, 25.43267343],
        [88.4323, 75.45146857], [18.52167173, 57.17151834], [96.87211957, 79.17682143],
        [90.88110424, 60.7095252, ], [2.85430696, 54.40447172], [99.91234515, 74.4014438, ],
        [73.94323878, 97.65276912], [37.71273758, 5.55845728]
    ]

    # --- Setup the Problem ---
    Covering.pts = pts
    Covering.side = 30  # Coverage radius of each antenna
    Covering.prepare_data()
    print(f"Problem: Cover {Covering.total_points_to_cover} points using the fewest antennas.\n")

    # --- Set BA Parameters ---
    ba_params = {
        'n': 300,  # Total population (3x the previous size for diversity)
        'e': 30,  # Elite sites (Keep at 10% of n)
        'm': 60,  # Best sites (Keep at 20% of n)
        'nep': 15,  # Foragers for elite (Strong local search)
        'nsp': 5,  # Foragers for best (Moderate local search)
        'max_iter': 1000  # Keep iterations high
    }

    # --- Run the Algorithm ---
    solver = BeeAlgorithm(**ba_params)
    solution = solver.solve()

    # --- Show the Result ---
    solution.draw(f"Best Solution Found: {solution.sm} Antennas, {solution.nb} Points Covered")