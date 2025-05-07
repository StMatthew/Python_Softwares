import tkinter as tk
import random
import time

class SnakeGame:
    def __init__(self, master):
        self.master = master
        master.title("Snazzy Snake Game!")
        master.config(bg="#222222")  # Dark background

        self.canvas_width = 600
        self.canvas_height = 400
        self.cell_size = 20
        self.delay = 0.1  # Time between game updates
        self.snake_color = "#8aff8a"  # Bright green
        self.food_color = "#ff6f69"  # Coral red
        self.bg_color = "#222222"

        self.canvas = tk.Canvas(master, width=self.canvas_width, height=self.canvas_height, bg=self.bg_color, highlightthickness=0)
        self.canvas.pack(pady=10)

        self.score = 0
        self.score_label = tk.Label(master, text="Score: 0", font=('Press Start 2P', 16), bg=self.bg_color, fg="#ffffff")
        self.score_label.pack()

        self.start_button = tk.Button(master, text="Start Game", font=('Press Start 2P', 14), bg="#4CAF50", fg="white", command=self.start_game, relief=tk.RAISED, bd=3, padx=20, pady=10)
        self.start_button.pack(pady=10)

        self.restart_button = tk.Button(master, text="Restart", font=('Press Start 2P', 14), bg="#f44336", fg="white", command=self.restart_game, relief=tk.RAISED, bd=3, padx=20, pady=10, state=tk.DISABLED)
        self.restart_button.pack(pady=5)

        self.game_running = False
        self.snake = []
        self.direction = ""
        self.food_position = None
        self.bind_keys()

    def setup_game(self):
        """Initial setup of the game."""
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = "Right"
        self.food_position = self.create_food()
        self.score = 0
        self.score_label.config(text="Score: 0")
        self.game_running = True
        self.update_game()

    def start_game(self):
        """Starts the game."""
        self.start_button.config(state=tk.DISABLED)
        self.restart_button.config(state=tk.NORMAL)
        self.canvas.delete("all")
        self.setup_game()

    def restart_game(self):
        """Restarts the game."""
        self.canvas.delete("all")
        self.setup_game()

    def create_food(self):
        """Creates food at a random empty location."""
        while True:
            x = random.randrange(0, self.canvas_width // self.cell_size) * self.cell_size
            y = random.randrange(0, self.canvas_height // self.cell_size) * self.cell_size
            if (x, y) not in self.snake:
                return (x, y)

    def move_snake(self):
        """Moves the snake in the current direction."""
        head_x, head_y = self.snake[0]

        if self.direction == "Right":
            new_head = (head_x + self.cell_size, head_y)
        elif self.direction == "Left":
            new_head = (head_x - self.cell_size, head_y)
        elif self.direction == "Up":
            new_head = (head_x, head_y - self.cell_size)
        elif self.direction == "Down":
            new_head = (head_x, head_y + self.cell_size)

        self.snake.insert(0, new_head)
        if new_head == self.food_position:
            self.food_position = self.create_food()
            self.score += 1
            self.score_label.config(text=f"Score: {self.score}")
        else:
            self.snake.pop()

    def check_collision(self):
        """Checks for collisions with walls or the snake's body."""
        head_x, head_y = self.snake[0]
        if not (0 <= head_x < self.canvas_width and 0 <= head_y < self.canvas_height) or self.snake[0] in self.snake[1:]:
            self.game_running = False
            self.game_over()

    def game_over(self):
        """Displays game over screen and asks to play again."""
        self.canvas.delete("all")
        self.canvas.create_text(self.canvas_width / 2, self.canvas_height / 2 - 30, text=f"Game Over!", font=('Press Start 2P', 28), fill="#ff4d4d")
        self.canvas.create_text(self.canvas_width / 2, self.canvas_height / 2 + 10, text=f"Score: {self.score}", font=('Press Start 2P', 18), fill="#ffffff")
        play_again_text = self.canvas.create_text(self.canvas_width / 2, self.canvas_height / 2 + 60, text="Play Again?", font=('Press Start 2P', 16), fill="#ffffff", cursor="hand2")
        self.canvas.tag_bind(play_again_text, "<Button-1>", lambda event: self.restart_game())

    def draw(self):
        """Draws the snake and food on the canvas with rounded corners."""
        self.canvas.delete("all")
        for segment in self.snake:
            x, y = segment
            self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, fill=self.snake_color, outline=self.bg_color, width=2) # Added outline for separation
        x_food, y_food = self.food_position
        radius = self.cell_size // 2
        self.canvas.create_oval(x_food, y_food, x_food + self.cell_size, y_food + self.cell_size, fill=self.food_color, outline=self.bg_color, width=1) # Nicer food

    def bind_keys(self):
        """Binds arrow keys to change snake direction."""
        self.master.bind("<Up>", lambda event: self.change_direction("Up"))
        self.master.bind("<Down>", lambda event: self.change_direction("Down"))
        self.master.bind("<Left>", lambda event: self.change_direction("Left"))
        self.master.bind("<Right>", lambda event: self.change_direction("Right"))

    def change_direction(self, new_direction):
        """Changes the snake's direction, preventing immediate 180-degree turns."""
        if new_direction == "Up" and self.direction != "Down":
            self.direction = "Up"
        elif new_direction == "Down" and self.direction != "Up":
            self.direction = "Down"
        elif new_direction == "Left" and self.direction != "Right":
            self.direction = "Left"
        elif new_direction == "Right" and self.direction != "Left":
            self.direction = "Right"

    def update_game(self):
        """Main game loop."""
        if self.game_running:
            self.move_snake()
            self.check_collision()
            self.draw()
            self.master.after(int(self.delay * 1000), self.update_game)

if __name__ == "__main__":
    root = tk.Tk()
    game = SnakeGame(root)
    root.mainloop()