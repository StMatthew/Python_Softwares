import tkinter as tk

class CalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Stylish Calculator")
        master.config(bg="#f0f0f0")  # Light grey background for the window

        self.expression = ""
        self.input_var = tk.StringVar()

        self.entry = tk.Entry(master, textvariable=self.input_var, width=25, bd=7, relief=tk.SUNKEN, font=('Segoe UI', 18), justify='right', bg="#e0e0e0", fg="black")
        self.entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

        button_data = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3, "#ffcc99"),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3, "#ffcc99"),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3, "#ffcc99"),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2, "#90ee90"), ('+', 4, 3, "#ffcc99")
        ]

        for (text, row, col, *color) in button_data:
            bg_color = color[0] if color else "#d3d3d3"  # Light grey default
            button = tk.Button(master, text=text, padx=25, pady=25, font=('Segoe UI', 16),
                               bg=bg_color, fg="black", activebackground="#a9a9a9",  # Dark grey when pressed
                               command=lambda t=text: self.button_click(t))
            button.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            master.grid_columnconfigure(col, weight=1)
            master.grid_rowconfigure(row, weight=1)

        # Configure row 0 separately for the entry
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)
        master.grid_columnconfigure(3, weight=1)

    def button_click(self, text):
        if text == "=":
            try:
                result = eval(self.expression)
                self.input_var.set(result)
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        else:
            self.expression += text
            self.input_var.set(self.expression)

if __name__ == "__main__":
    root = tk.Tk()
    calculator = CalculatorGUI(root)
    root.mainloop()