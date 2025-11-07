import tkinter as tk  # Import tkinter for visualization

class GridVisualizer:
    def __init__(self, grid_template, solution_path_object, cell_size=40, anim_speed=0.1):
        """
        Initializes the visualizer.

        Args:
            grid_template (list[list[int]]): The original benchmark grid.
            solution_path_object (PathSolution): The *single* best solution
                                                 returned by the solver.
            cell_size (int): Size of each cell in pixels.
            anim_speed (float): Seconds to wait between drawing each path segment.
        """
        self.grid_template = grid_template
        self.solution_path_coords = solution_path_object.path_coords
        self.cell_size = cell_size
        self.anim_speed = anim_speed
        self.rows = len(grid_template)
        self.cols = len(grid_template[0])

        self.root = tk.Tk()
        self.root.title(f"Bee Algorithm Solution (Fitness: {len(self.solution_path_coords)})")

        self.canvas = tk.Canvas(self.root, width=self.cols * cell_size, height=self.rows * cell_size)
        self.canvas.pack()

        self.draw_initial_grid()

        # Start the animation after a short delay
        self.root.after(500, self.animate_solution)
        self.root.mainloop()

    def draw_initial_grid(self):
        """Draws the grid *before* the path is animated."""
        self.canvas.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                color = "black"  # 0 = Black (wall)
                if self.grid_template[r][c] == 1:
                    color = "blue"  # 1 = Blue (unvisited)

                self.canvas.create_rectangle(
                    c * self.cell_size, r * self.cell_size,
                    (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                    fill=color, outline="white"
                )

    def animate_solution(self, step=0):
        """
        Animates the path by drawing one segment at a time.
        """
        if step < len(self.solution_path_coords):
            # Get the coordinate for the current step
            r, c = self.solution_path_coords[step]

            # Draw the green square for this step
            self.canvas.create_rectangle(
                c * self.cell_size, r * self.cell_size,
                (c + 1) * self.cell_size, (r + 1) * self.cell_size,
                fill="green", outline="white"
            )

            # Schedule the next step of the animation
            self.root.after(int(self.anim_speed * 1000), self.animate_solution, step + 1)