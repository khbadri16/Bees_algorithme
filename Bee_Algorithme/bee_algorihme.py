import copy
import random
import time
from representation import GridVisualizer


# --- The "Food Source" / "Solution" Class ---
class PathSolution:
    """
    Represents a single "Food Source" or "Solution" for the Bee Algorithm.
    A solution is a complete path of visited blue squares.
    Its "fitness" is the length of that path.
    """

    def __init__(self, grid_template, start_pos):
        """
        Initializes the path with a given grid and a starting position.

        Args:
            grid_template (list[list[int]]): The master grid (0s and 1s).
            start_pos (tuple): The (row, col) starting coordinate.
        """
        # Ensure start position is valid (must be a '1')
        if grid_template[start_pos[0]][start_pos[1]] == 0:
            raise ValueError("Start position must be on a blue square (1).")

        self.grid_template = grid_template
        # Create a unique board for this path to draw on
        self.board = copy.deepcopy(grid_template)
        self.start_pos = start_pos

        # The path is a list of (row, col) coordinates
        self.path_coords = [start_pos]

        # Mark the start position as visited (2 = green)
        self.board[start_pos[0]][start_pos[1]] = 2

        # Fitness is the number of squares visited (path length)
        self.fitness = 1

        self.rows = len(grid_template)
        self.cols = len(grid_template[0])

    def _get_valid_moves(self, current_pos):
        """
        Finds all adjacent, unvisited blue squares (1s).
        This is the same logic as your original 'actions' method.
        """
        r, c = current_pos
        moves = []
        # (dr, dc) = (Up, Down, Left, Right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            # Check boundaries and if it's an unvisited blue square (1)
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.board[nr][nc] == 1:
                moves.append((nr, nc))
        return moves

    def perform_random_walk(self):
        """
        This is the "Scout Bee" logic.
        It starts from self.start_pos and takes random valid moves
        until it gets stuck.
        """
        current_pos = self.start_pos
        while True:
            valid_moves = self._get_valid_moves(current_pos)

            # If no moves, the path is complete (stuck)
            if not valid_moves:
                break

                # Pick a move randomly
            next_pos = random.choice(valid_moves)

            # Take the move
            self.path_coords.append(next_pos)
            self.board[next_pos[0]][next_pos[1]] = 2  # Mark as visited (green)
            current_pos = next_pos

        # Final fitness is the total length of the path
        self.fitness = len(self.path_coords)

    def perform_mutated_walk(self, base_path_coords, mutation_point):
        """
        This is the "Forager Bee" (Local Search) logic.
        It follows a good 'base_path' for a while, then
        "mutates" by starting a new random walk from that point.
        """
        try:
            # 1. Follow the base path up to the mutation point
            for i in range(1, mutation_point):
                pos = base_path_coords[i]
                # We must check if this move is valid *on our own board*
                # in case the base path had an invalid crossover.
                if self.board[pos[0]][pos[1]] == 1:
                    self.path_coords.append(pos)
                    self.board[pos[0]][pos[1]] = 2  # Mark as visited
                else:
                    # Path is invalid, stop following
                    break

            current_pos = self.path_coords[-1]

            # 2. Start a new random walk from that point
            while True:
                valid_moves = self._get_valid_moves(current_pos)
                if not valid_moves:
                    break  # Stuck

                next_pos = random.choice(valid_moves)

                self.path_coords.append(next_pos)
                self.board[next_pos[0]][next_pos[1]] = 2
                current_pos = next_pos

            self.fitness = len(self.path_coords)
        except Exception as e:
            # This can happen if the base_path is no longer valid
            # In this case, the path is just a poor-fitness one.
            self.fitness = len(self.path_coords)

    def __str__(self):
        """Utility function to print the solved grid."""
        # 0=Black(Block), 1=Blue(Not Visited), 2=Green(Visited Path)
        chars = " #~"
        s = f"Path Length (Fitness): {self.fitness}\n"
        for r in range(self.rows):
            s += "".join([f"[{chars[self.board[r][c]]}]" for c in range(self.cols)]) + "\n"
        return s

    # For sorting
    def __lt__(self, other):
        return self.fitness < other.fitness


# --- The Bee Algorithm Solver ---

class BeeAlgorithmSolver:

    def __init__(self, grid_template, start_pos, n, e, m, nep, nsp, max_iterations):
        """
        Initializes the Bee Algorithm solver with all parameters.

        Args:
            grid_template (list[list[int]]): The 12x12 grid.
            start_pos (tuple): The (row, col) starting point.
            n (int): Total population (number of scout bees).
            e (int): Number of "Elite" sites.
            m (int): Number of "Best" (non-elite) sites.
            nep (int): Number of forager bees for elite sites.
            nsp (int): Number of forager bees for best sites.
            max_iterations (int): Global stopping condition.
        """
        self.grid_template = grid_template
        self.start_pos = start_pos

        # BA Params
        self.n = n  # Total population
        self.e = e  # Elite sites
        self.m = m  # Best sites (non-elite)
        self.nep = nep  # Foragers for elite
        self.nsp = nsp  # Foragers for best
        self.max_iterations = max_iterations

        self.population = []
        self.global_best_path = None

        # Calculate the "perfect" score
        self.goal_fitness = sum(row.count(1) for row in grid_template)
        print(f"Goal: Visit all {self.goal_fitness} blue squares.")
        print(f"Start Position: {start_pos}\n")

        if self.e + self.m > self.n:
            raise ValueError("Population (n) must be larger than elite (e) + best (m).")

    def _generate_random_path(self):
        """Creates one new Scout Bee solution."""
        path = PathSolution(self.grid_template, self.start_pos)
        path.perform_random_walk()
        return path

    def _local_search(self, base_path, num_foragers):
        """
        This is the "Forager Bee" (Neighborhood Search) logic.
        It creates 'num_foragers' new paths by "mutating" the base_path
        and returns the best one it finds.
        """
        # The current best is the base_path itself
        best_neighbor = base_path

        for _ in range(num_foragers):
            # Pick a random point to "mutate" from.
            # We'll pick a point somewhere in the middle of the path.
            path_len = len(base_path.path_coords)
            if path_len <= 2:
                mutation_point = 1
            else:
                # Mutate from somewhere in the first 3/4 of the path
                mutation_point = random.randint(1, max(1, int(path_len * 0.75)))

            # Create a new bee (path)
            new_path = PathSolution(self.grid_template, self.start_pos)
            new_path.perform_mutated_walk(base_path.path_coords, mutation_point)

            # If this new "neighbor" path is better, it's the new best
            if new_path.fitness > best_neighbor.fitness:
                best_neighbor = new_path

        return best_neighbor

    def solve(self):
        """
        Runs the main Bee Algorithm loop.
        """
        start_time = time.time()

        # 1. INITIALIZATION PHASE (All bees are scouts)
        print("--- Initializing Scout Bees ---")
        self.population = [self._generate_random_path() for _ in range(self.n)]

        # Initialize global best (The "Memory"
        self.population.sort(key=lambda p: p.fitness, reverse=True)
        self.global_best_path = self.population[0]
        print(f"Initial best fitness: {self.global_best_path.fitness}")

        # 2. MAIN LOOP (Repeat for max_iterations)
        final_iter = 0
        for i in range(self.max_iterations):
            final_iter = i + 1

            # 2a. RANK & SELECT
            self.population.sort(key=lambda p: p.fitness, reverse=True)

            # 2b. UPDATE GLOBAL BEST (Memory)
            # Check if the best bee in the *current* population
            # is better than the best *ever* found.
            if self.population[0].fitness > self.global_best_path.fitness:
                self.global_best_path = self.population[0]

            # Print status
            if (i + 1) % 50 == 0:
                print(
                    f"Iter {i + 1}/{self.max_iterations} | Best Fitness So Far: {self.global_best_path.fitness}/{self.goal_fitness}")

            # 2c. CHECK FOR GOAL
            if self.global_best_path.fitness == self.goal_fitness:
                print(f"\n--- GOAL ACHIEVED! ---")
                break

            next_population = []

            # 2d. RECRUITMENT (Elite Sites Local Search)
            for j in range(self.e):
                base_path = self.population[j]
                best_elite_neighbor = self._local_search(base_path, self.nep)
                next_population.append(best_elite_neighbor)

            # 2e. RECRUITMENT (Best Sites Local Search)
            for j in range(self.e, self.e + self.m):
                base_path = self.population[j]
                best_neighbor = self._local_search(base_path, self.nsp)
                next_population.append(best_neighbor)

            # 2f. GLOBAL SEARCH (New Scouts / Site Abandonment)
            # The 'n - e - m' worst sites are abandoned.
            # They are replaced by new, random scout bees.
            num_scouts = self.n - self.e - self.m
            for _ in range(num_scouts):
                next_population.append(self._generate_random_path())

            # The new generation is complete
            self.population = next_population

        # 3. END OF ALGORITHM
        end_time = time.time()
        print("\n--- Algorithm Finished ---")
        print(f"Total iterations: {final_iter}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"\nBest solution found (Fitness: {self.global_best_path.fitness}/{self.goal_fitness}):")

        # Print the best path found
        print(self.global_best_path)
        return self.global_best_path


# --- Main Execution ---

if __name__ == "__main__":
    # The 12x12 grid from your benchmark
    benchmark1 =  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
                  [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]

    # A valid starting position (must be a '1')
    # (5,5) is a '0'. Let's pick (0,1)
    START_POINT = (3, 3)

    # --- Bee Algorithm Parameters (You can tune these!) ---

    # Total population of bees (solutions)
    N_POPULATION = 100

    # Number of "Elite" sites (the very best)
    E_ELITE_SITES = 20

    # Number of "Best" sites (the good ones)
    M_BEST_SITES = 10

    # Number of foragers sent to ELITE sites (strong local search)
    NEP_ELITE_FORAGERS = 5

    # Number of foragers sent to BEST sites (weaker local search)
    NSP_BEST_FORAGERS = 2

    # Global stop condition
    MAX_ITERATIONS = 8000

    # --- Run the Solver ---
    solver = BeeAlgorithmSolver(
        grid_template=benchmark1,
        start_pos=START_POINT,
        n=N_POPULATION,
        e=E_ELITE_SITES,
        m=M_BEST_SITES,
        nep=NEP_ELITE_FORAGERS,
        nsp=NSP_BEST_FORAGERS,
        max_iterations=MAX_ITERATIONS
    )

    best_path_found = solver.solve()
    if best_path_found:
        print("\nLaunching visualizer...")
        # We pass the *original grid* and the *final path object*
        GridVisualizer(benchmark1, best_path_found, cell_size=30, anim_speed=0.05)
    else:
        print("No solution was found to visualize.")
