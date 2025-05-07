import tkinter as tk
import math

class ScientificCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Scientific Calculator")
        master.config(bg="#f0f0f0")

        self.expression = ""
        self.input_var = tk.StringVar()

        self.entry = tk.Entry(master, textvariable=self.input_var, width=40, bd=7, relief=tk.SUNKEN, font=('Segoe UI', 18), justify='right', bg="#e0e0e0", fg="black")
        self.entry.grid(row=0, column=0, columnspan=5, padx=10, pady=10)

        button_data = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3, "#ffcc99"), ('sqrt', 1, 4, "#a0d468"),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3, "#ffcc99"), ('^', 2, 4, "#a0d468"),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3, "#ffcc99"), ('sin', 3, 4, "#a0d468"),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2, "#90ee90"), ('+', 4, 3, "#ffcc99"), ('cos', 4, 4, "#a0d468"),
            ('C', 5, 0, "#f44336"), ('(', 5, 1), (')', 5, 2), ('log', 5, 3, "#a0d468"), ('tan', 5, 4, "#a0d468"),
            ('ln', 6, 0, "#a0d468"), ('%', 6, 1), ('!', 6, 2, "#a0d468"), ('1/x', 6, 3, "#a0d468"), ('pi', 6, 4, "#a0d468")
        ]

        for (text, row, col, *color) in button_data:
            bg_color = color[0] if color else "#d3d3d3"
            button = tk.Button(master, text=text, padx=20, pady=20, font=('Segoe UI', 14),
                               bg=bg_color, fg="black", activebackground="#a9a9a9",
                               command=lambda t=text: self.button_click(t))
            button.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            master.grid_columnconfigure(col, weight=1)
            master.grid_rowconfigure(row, weight=1)

        for i in range(5):
            master.grid_columnconfigure(i, weight=1)

    def format_fraction(self, numerator, denominator):
        """Formats a fraction as a visually distinct string."""
        return f"{numerator}\u2044{denominator}"  # Using Unicode fraction slash

    def format_mixed_fraction(self, whole, numerator, denominator):
        """Formats a mixed fraction."""
        return f"{whole} {numerator}\u2044{denominator}"

    def evaluate_expression(self):
        """Evaluates the expression and formats the result."""
        try:
            result = eval(self.expression)
            if isinstance(result, float):
                # Try to represent as a fraction if it's a simple rational number
                from fractions import Fraction
                fraction_result = Fraction(result).limit_denominator()
                if fraction_result.denominator != 1:
                    whole = 0
                    num = fraction_result.numerator
                    den = fraction_result.denominator
                    if abs(num) >= den:
                        whole = num // den
                        num %= den
                    if whole == 0:
                        return self.format_fraction(num, den)
                    else:
                        return self.format_mixed_fraction(whole, abs(num), den)
            return str(result)
        except Exception as e:
            return "Error"

    def button_click(self, text):
        if text == "=":
            self.input_var.set(self.evaluate_expression())
            self.expression = self.input_var.get() # Update expression with the result
        elif text == "C":
            self.expression = ""
            self.input_var.set("")
        elif text == "sqrt":
            try:
                result = math.sqrt(eval(self.expression))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "^":
            self.expression += "**"
            self.input_var.set(self.expression)
        elif text == "sin":
            try:
                result = math.sin(math.radians(eval(self.expression)))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "cos":
            try:
                result = math.cos(math.radians(eval(self.expression)))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "tan":
            try:
                result = math.tan(math.radians(eval(self.expression)))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "log":
            try:
                result = math.log10(eval(self.expression))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "ln":
            try:
                result = math.log(eval(self.expression))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "!":
            try:
                result = math.factorial(int(eval(self.expression)))
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "1/x":
            try:
                result = 1 / eval(self.expression)
                self.input_var.set(self.format_result(result))
                self.expression = str(result)
            except Exception as e:
                self.input_var.set("Error")
                self.expression = ""
        elif text == "pi":
            self.expression += str(math.pi)
            self.input_var.set(self.expression)
        elif text == "%":
            self.expression += "/100*"
            self.input_var.set(self.expression)
        else:
            self.expression += text
            self.input_var.set(self.expression)

    def format_result(self, result):
        """Attempts to format the result as a fraction or mixed fraction."""
        if isinstance(result, float):
            from fractions import Fraction
            fraction_result = Fraction(result).limit_denominator()
            if fraction_result.denominator != 1:
                whole = 0
                num = fraction_result.numerator
                den = fraction_result.denominator
                if abs(num) >= den:
                    whole = num // den
                    num %= den
                if whole == 0:
                    return self.format_fraction(num, den)
                else:
                    return self.format_mixed_fraction(whole, abs(num), den)
        return str(result)

if __name__ == "__main__":
    root = tk.Tk()
    calculator = ScientificCalculatorGUI(root)
    root.mainloop()