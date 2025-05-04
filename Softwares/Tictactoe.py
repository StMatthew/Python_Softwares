import tkinter as tk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, master):
        self.master = master
        master.title("Tic-Tac-Toe!")
        master.config(bg="white")  # White background

        self.current_player = "X"
        self.board = [""] * 9
        self.buttons = []
        self.font_x = ('Arial', 60, 'bold')
        self.font_o = ('Arial', 60)
        self.grid_color = "black"
        self.cell_bg = "white"
        self.game_active = True

        for i in range(9):
            row = i // 3
            col = i % 3
            button = tk.Button(
                master,
                text="",
                font=self.font_x if self.current_player == "X" else self.font_o,
                width=2,
                height=1,
                bg=self.cell_bg,
                fg=self.grid_color,
                command=lambda index=i: self.button_click(index),
                relief=tk.SOLID,
                bd=2,
                highlightbackground=self.grid_color,
                highlightcolor=self.grid_color
            )
            button.grid(row=row, column=col, padx=0, pady=0, sticky="nsew")
            self.buttons.append(button)
            master.grid_columnconfigure(col, weight=1)
            master.grid_rowconfigure(row, weight=1)

        reset_button = tk.Button(
            master,
            text="Reset Game",
            font=('Arial', 16),
            bg="#4CAF50",
            fg="white",
            command=self.reset_game,
            relief=tk.RAISED,
            bd=3,
            padx=10,
            pady=5
        )
        reset_button.grid(row=3, column=0, columnspan=3, padx=5, pady=10, sticky="ew")

    def button_click(self, index):
        if self.game_active and self.board[index] == "":
            self.board[index] = self.current_player
            self.buttons[index].config(text=self.current_player, font=self.font_x if self.current_player == "X" else self.font_o, fg=self.grid_color)
            if self.check_win():
                self.game_over(f"Player {self.current_player} wins!")
            elif self.check_draw():
                self.game_over("It's a draw!")
            else:
                self.switch_player()

    def switch_player(self):
        self.current_player = "O" if self.current_player == "X" else "X"
        self.master.title(f"Tic-Tac-Toe! - Current Player: {self.current_player}")

    def check_win(self):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]             # Diagonals
        ]
        for condition in win_conditions:
            if self.board[condition[0]] == self.board[condition[1]] == self.board[condition[2]] != "":
                self.game_active = False
                return True
        return False

    def check_draw(self):
        if all(cell != "" for cell in self.board):
            self.game_active = False
            return True
        return False

    def game_over(self, message):
        messagebox.showinfo("Game Over", message)
        self.reset_game()

    def reset_game(self):
        self.current_player = "X"
        self.board = [""] * 9
        self.game_active = True
        self.master.title("Tic-Tac-Toe!")
        for button in self.buttons:
            button.config(text="", font=self.font_x, fg=self.grid_color)

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToe(root)
    root.mainloop()