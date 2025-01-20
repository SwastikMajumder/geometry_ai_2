import inference_engine
import spatial_construction

command = """draw quadrilateral
join AC
join BD
equation angle_val ABC 90
equation line_eq AD CD
equation line_eq CD BC
equation line_eq BC AB
compute"""

from PIL import Image, ImageDraw, ImageFont, ImageTk
import tkinter as tk
from tkinter import Text, Scrollbar
import threading

class Space:
    def __init__(self):
        self.points = []
        self.point_pairs = []
        self.perpendicular_angle_list = []

space = Space()
def print_text(text, color="black", auto_next_line=True):
    if auto_next_line:
        console.insert(tk.END, text + "\n", color)
    else:
        console.insert(tk.END, text, color)
    console.see(tk.END)

def a2n(letter):
    return ord(letter) - ord("A")

def a2n2(line):
    return (a2n(line[0]), a2n(line[1]))

def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


def print_diagram(space):
    lines = [(space.points[start], space.points[end]) for start, end in space.point_pairs]
    image = spatial_construction.draw_points_and_lines(space.points, lines)
    image.save("points_and_lines_image.png")
    display_image("points_and_lines_image.png")
    
def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


def run_parallel_function():
    global space
    global command
    count = 0
    command = command.split("\n")
    command_new = []
    string = None
    while True:
        print_diagram(space)
        
        if command != []:
            if command[0].split(" ")[0] == "equation":
                command_new.append(command[0])
            string = command.pop(0)
            print_text(">>> ", "green", False)
            print_text(string, "blue", True)
        else:
            
            command = inference_engine.compute(space, "\n".join(command_new))
            for string in command:
                print_text(string, "black", True)
            print_text("\nend of program", "green", True)
            return
        if string[:13] == "draw triangle":
            space = spatial_construction.draw_triangle(space)
        elif string.split(" ")[0] == "normal" and string.split(" ")[2] == "on":
            p = string.split(" ")[1]
            line = string.split(" ")[3]
            normal_point_fraction(
                spatial_construction.points[a2n(line[0])], spatial_construction.points[a2n(line[0])], spatial_construction.points[a2n(p)]
            )
        elif string == "draw quadrilateral":
            space = spatial_construction.draw_quadrilateral(space)

        elif string.split(" ")[0] == "perpendicular" and string.split(" ")[2] == "to":
            space = spatial_construction.perpendicular(space, string.split(" ")[1], string.split(" ")[3])
            
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "from":
            space = spatial_construction.extend(
                space, string.split(" ")[1], string.split(" ")[3], 200
            )
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "to":
            val = spatial_construction.find_intersection(
                space.points[a2n(string.split(" ")[1][0])][0],
                space.points[a2n(string.split(" ")[1][0])][1],
                space.points[a2n(string.split(" ")[1][1])][0],
                space.points[a2n(string.split(" ")[1][1])][1],
                space.points[a2n(string.split(" ")[3][0])][0],
                space.points[a2n(string.split(" ")[3][0])][1],
                space.points[a2n(string.split(" ")[3][1])][0],
                space.points[a2n(string.split(" ")[3][1])][1],
            )
            space = spatial_construction.divide_line(space, string.split(" ")[3], val[0])
        elif string.split(" ")[0] == "split":
            space = spatial_construction.divide_line(space, string.split(" ")[-1])
        elif string.split(" ")[0] == "join":
            space = spatial_construction.connect_point(space, string.split(" ")[1])


root = tk.Tk()
root.title("Geometry Ai")
root.resizable(False, False)

console_frame = tk.Frame(root)
console_frame.grid(row=0, column=0, padx=10, pady=10)

scrollbar = Scrollbar(console_frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

console = Text(console_frame, width=40, height=20, yscrollcommand=scrollbar.set)
console.pack(side=tk.LEFT)

scrollbar.config(command=console.yview)

console.tag_configure("black", foreground="black")
console.tag_configure("blue", foreground="blue")
console.tag_configure("red", foreground="red")
console.tag_configure("green", foreground="green")

image_frame = tk.Frame(root)
image_frame.grid(row=0, column=1, padx=10, pady=10)

image_label = tk.Label(image_frame)
image_label.pack(pady=5)

parallel_thread = threading.Thread(target=run_parallel_function, daemon=True)
parallel_thread.start()

root.mainloop()
