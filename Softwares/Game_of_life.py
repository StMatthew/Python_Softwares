import tkinter as tk
import random
import time

class GameOfLifeGUI:
    def __init__(self, master, rows=50, cols=50, cell_size=10, update_interval=0.1):
        self.master = master
        master.title("Conway's Game of Life")

        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.update_interval = update_interval
        self.grid = self.create_grid()

        self.canvas = tk.Canvas(master, width=cols * cell_size, height=rows * cell_size, bg="white")
        self.canvas.pack()

        self.running = False
        self.start_time = 0
        self.duration = 300  # 5 minutes in seconds

        start_button = tk.Button(master, text="Start", command=self.start_game)
        start_button.pack(side=tk.LEFT)

        pause_button = tk.Button(master, text="Pause", command=self.pause_game)
        pause_button.pack(side=tk.LEFT)

        randomize_button = tk.Button(master, text="Randomize", command=self.randomize_grid)
        randomize_button.pack(side=tk.LEFT)

        self.draw_grid()

    def create_grid(self):
        """Creates a 2D grid with random live or dead cells."""
        return [[random.choice([0, 1]) for _ in range(self.cols)] for _ in range(self.rows)]

    def randomize_grid(self):
        """Randomizes the current grid."""
        self.grid = self.create_grid()
        self.draw_grid()

    def get_neighbors(self, row, col):
        """Counts the number of live neighbors for a given cell."""
        neighbors = 0
        for i in range(max(0, row - 1), min(self.rows, row + 2)):
            for j in range(max(0, col - 1), min(self.cols, col + 2)):
                if (i, j) != (row, col) and self.grid[i][j] == 1:
                    neighbors += 1
        return neighbors

    def next_generation(self):
        """Calculates the next generation of the grid."""
        new_grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        for row in range(self.rows):
            for col in range(self.cols):
                live_neighbors = self.get_neighbors(row, col)
                if self.grid[row][col] == 1:
                    if live_neighbors < 2 or live_neighbors > 3:
                        new_grid[row][col] = 0
                    else:
                        new_grid[row][col] = 1
                else:
                    if live_neighbors == 3:
                        new_grid[row][col] = 1
        self.grid = new_grid
        self.draw_grid()

    def draw_grid(self):
        """Draws the current state of the grid on the canvas."""
        self.canvas.delete("all")
        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = "black" if self.grid[row][col] == 1 else "white"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="grey")

    def update_game(self):
        """Updates the game state and redraws the grid."""
        if self.running and time.time() - self.start_time < self.duration:
            self.next_generation()
            self.master.after(int(self.update_interval * 1000), self.update_game)
        elif self.running:
            self.running = False
            print("Game ended after 5 minutes.")

    def start_game(self):
        """Starts the game simulation."""
        if not self.running:
            self.running = True
            self.start_time = time.time()
            self.update_game()

    def pause_game(self):
        """Pauses the game simulation."""
        self.running = False

if __name__ == "__main__":
    root = tk.Tk()
    game = GameOfLifeGUI(root)
    root.mainloop()