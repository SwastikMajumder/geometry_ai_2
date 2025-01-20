import anglegen13
command = """draw quadrilateral
join AC
join BD
equation angle_val ABC 90
equation line_eq AD CD
equation line_eq CD BC
equation line_eq BC AB
compute"""

import re
import itertools
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from fractions import Fraction
import sys
import tkinter as tk
from tkinter import Text, Scrollbar
import copy
import threading
import time

points = []
point_pairs = []
perpendicular_angle_list = []

class TreeNode:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []


def tree_form(tabbed_strings):
    lines = tabbed_strings.split("\n")
    root = TreeNode("Root")
    current_level_nodes = {0: root}
    stack = [root]
    for line in lines:
        level = line.count(" ")
        node_name = line.strip()
        node = TreeNode(node_name)
        while len(stack) > level + 1:
            stack.pop()
        parent_node = stack[-1]
        parent_node.children.append(node)
        current_level_nodes[level] = node
        stack.append(node)
    return root.children[0]


def str_form(node):
    def recursive_str(node, depth=0):
        result = "{}{}".format(" " * depth, node.name)
        for child in node.children:
            result += "\n" + recursive


def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    try:
        if x1 == x2:
            if x3 == x4:
                return None, "parallel vertical lines"
            m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None
            x = x1
            y = m2 * x + (y3 - m2 * x3) if m2 is not None else None
            return (x, y), "intersect" if y is not None else "no intersection"

        if x3 == x4:
            m1 = (y2 - y1) / (x2 - x1)
            x = x3
            y = m1 * x + (y1 - m1 * x1)
            return (x, y), "intersect"

        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else None
        m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else None

        if m1 == m2:
            return None, "parallel lines"

        if m1 is None:
            x = x1
            y = m2 * x + (y3 - m2 * x3)
        elif m2 is None:
            x = x3
            y = m1 * x + (y1 - m1 * x1)
        else:

            a = m1
            b = y1 - m1 * x1
            c = m2
            d = y3 - m2 * x3
            x = (d - b) / (a - c)
            y = a * x + b

        return (x, y), "intersect"
    except:
        return None, "error"


def find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4):

    if x2 == x1 and x4 == x3:
        return None, "error"
    elif x2 == x1:
        x = x1
        m2 = (y4 - y3) / (x4 - x3)
        d = y3 - m2 * x3
        y = m2 * x + d
    elif x4 == x3:
        x = x3
        m1 = (y2 - y1) / (x2 - x1)
        b = y1 - m1 * x1
        y = m1 * x + b
    else:

        m1 = (y2 - y1) / (x2 - x1)
        m2 = (y4 - y3) / (x4 - x3)

        if m1 == m2:
            return None, "error"

        a = m1
        b = y1 - m1 * x1
        c = m2
        d = y3 - m2 * x3
        x = (d - b) / (a - c)
        y = a * x + b

    def is_within(x1, x2, x):
        return min(x1, x2) <= x <= max(x1, x2)

    if (
        is_within(x1, x2, x)
        and is_within(y1, y2, y)
        and is_within(x3, x4, x)
        and is_within(y3, y4, y)
    ):
        return (x, y), "intersect"
    return None, "error"


def find_intersections_2(points, point_pairs):

    intersections = []
    for item in itertools.combinations(point_pairs, 2):
        x1, y1 = points[item[0][0]]
        x2, y2 = points[item[0][1]]
        x3, y3 = points[item[1][0]]
        x4, y4 = points[item[1][1]]
        tmp = find_intersection_3(x1, y1, x2, y2, x3, y3, x4, y4)
        if tmp[1] == "intersect":
            intersections.append((tmp[0], item))

    filtered_intersections = [point for point in intersections if point[0] not in points]

    return filtered_intersections


def a2n(letter):
    return ord(letter) - ord("A")


def a2n2(line):
    return (a2n(line[0]), a2n(line[1]))


def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    def is_within(x1, x2, x):
        return min(x1, x2) <= x <= max(x1, x2)

    ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
    output = False
    if (
        ans[1] == "intersect"
        and is_within(x3, x4, ans[0][0])
        and is_within(y3, y4, ans[0][1])
    ):
        output = True

    return output


def polygon_area(points):
    n = len(points)
    area = Fraction(0)
    for i in range(n - 1):
        area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
    area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
    return abs(area) / 2


def surrounding_angle(given_point):
    def is_enclosed_angle(curr, h1, h2, h3):
        return find_intersection_line_with_segment(
            curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1]
        )

    lst = []
    for line in point_pairs:
        if given_point == line[0]:
            lst.append(points[line[1]])
        elif given_point == line[1]:
            lst.append(points[line[0]])

    for item in itertools.permutations(lst):
        if all(
            is_enclosed_angle(points[given_point], item[i], item[i + 1], item[i + 2])
            for i in range(0, len(item) - 2, 1)
        ):
            lst = list(item)
            break

    tmp = [points.index(x) for x in lst]

    return tmp


def n2a(number):
    return chr(number + ord("A"))


def straight_line_2(point_list):
    global lines
    global points
    global point_pairs
    point_list = [a2n(x) for x in point_list]
    tmp = polygon_area([points[x] for x in point_list])
    return tmp == Fraction(0)


def straight_line(point_list):
    global lines
    global points
    global point_pairs
    tmp = polygon_area([points[x] for x in point_list])

    return tmp == Fraction(0)


def draw_points_and_lines(
    points,
    lines,
    image_size=(2000, 2000),
    point_radius=5,
    point_color=(0, 0, 0),
    line_color=(255, 0, 0),
    text_color=(0, 0, 0),
):

    image = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 72)
    except IOError:
        font = ImageFont.load_default()

    for index, (x, y) in enumerate(points):
        draw.ellipse(
            (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
            fill=point_color,
        )

        draw.text(
            (x + point_radius + 5, y - point_radius - 5),
            n2a(index),
            fill=text_color,
            font=font,
        )

    for (x1, y1), (x2, y2) in lines:
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=2)

    return image


def print_text(text, color="black", force=False, auto_next_line=True):
    global print_on
    if not force and not print_on:
        return

    if auto_next_line:
        console.insert(tk.END, text + "\n", color)
    else:
        console.insert(tk.END, text, color)
    console.see(tk.END)


def display_image(image_path):

    img = Image.open(image_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


def print_diagram():
    global points
    global point_pairs
    lines = [(points[start], points[end]) for start, end in point_pairs]
    image = draw_points_and_lines(points, lines)

    image.save("points_and_lines_image.png")

    display_image("points_and_lines_image.png")


def travel_till_end(start, step):
    global points
    global point_pairs
    done = False
    step_taken = [step]
    while not done:
        done = True
        for item in surrounding_angle(step):
            if (
                straight_line([step, start, item])
                and item not in step_taken
                and start != item
                and step != start
                and step != item
            ):
                step_taken.append(item)
                step = item

                done = False
                break
    return step


def sur(angle):
    global points
    global point_pairs
    count = 0
    if a2n(angle[0]) in surrounding_angle(a2n(angle[1])):
        count += 1
    if a2n(angle[2]) in surrounding_angle(a2n(angle[1])):
        count += 1
    return count


def print_angle(a, b, c, a_do=True, c_do=True):
    global points
    global point_pairs

    if a_do:
        a = travel_till_end(b, a)
    else:
        a = travel_till_end(b, a)
        a = travel_till_end(b, a)
    if c_do:
        c = travel_till_end(b, c)
    else:
        c = travel_till_end(b, c)
        c = travel_till_end(b, c)

    m, n = sorted([a, c])
    return n2a(m) + n2a(b) + n2a(n)


def print_angle_2(angle, a_do=True, c_do=True):
    global points
    global point_pairs
    x = angle
    return print_angle(a2n(x[0]), a2n(x[1]), a2n(x[2]), a_do, c_do)


def print_angle_3(angle):
    lst = [
        print_angle_2(angle, True, True),
        print_angle_2(angle, True, False),
        print_angle_2(angle, False, True),
        print_angle_2(angle, False, False),
    ]
    return sorted(lst, key=lambda x: sur(x))[0]


def print_angle_4(a, b, c):
    return print_angle_3(n2a(a) + n2a(b) + n2a(c))


def combine(a, b):
    global lines
    global points
    global point_pairs

    a = print_angle_3(a)
    b = print_angle_3(b)
    if a[1] != b[1]:
        return None
    if len(set(a + b)) != 4:
        return None
    r = a[0] + a[2] + b[0] + b[2]
    r = r.replace([x for x in r if r.count(x) == 2][0], "")
    out = print_angle_3(r[0] + b[1] + r[1])

    return out


def angle_sort(angle):
    if a2n(angle[0]) > a2n(angle[2]):
        angle = angle[2] + angle[1] + angle[0]
    return angle


def line_sort(line):
    if a2n(line[0]) > a2n(line[1]):
        line = line[1] + line[0]
    return line


def normal_point_fraction(A, B, P, alpha=Fraction(500)):
    global points
    x1, y1 = A
    x2, y2 = B
    x3, y3 = P

    v_x = x2 - x1
    v_y = y2 - y1

    if y1 == y2:
        normal_x = Fraction(0)
        normal_y = alpha
        new_x = x3 + normal_x
        new_y = y3 + normal_y
        points.append((new_x, new_y))
        return (new_x, new_y)

    if x1 == x2:
        normal_x = alpha
        normal_y = Fraction(0)
        new_x = x3 + normal_x
        new_y = y3 + normal_y
        points.append((new_x, new_y))
        return (new_x, new_y)

    normal_x = y1 - y2
    normal_y = x2 - x1

    length_squared = normal_x**2 + normal_y**2

    normal_x = normal_x * alpha / length_squared
    normal_y = normal_y * alpha / length_squared

    new_x = x3 + normal_x
    new_y = y3 + normal_y

    points.append((new_x, new_y))


def perpendicular_line_intersection(segment_start, segment_end, point):
    x1, y1 = segment_start
    x2, y2 = segment_end
    xp, yp = point

    if x2 == x1:
        xq = x1
        yq = yp

    elif y2 == y1:
        xq = xp
        yq = y1

    else:

        m = (y2 - y1) / (x2 - x1)

        m_perp = -1 / m

        xq = (m * x1 - m_perp * xp + yp - y1) / (m - m_perp)

        yq = m * (xq - x1) + y1

    return (xq, yq)


def extend(line, point_start, distance):
    global points
    b = None
    a = points[a2n(point_start)]
    if line[0] == point_start:
        b = points[a2n(line[1])]
    else:
        b = points[a2n(line[0])]
    ba = [a[0] - b[0], a[1] - b[1]]
    length_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    unit_vector_ba = [ba[0] / length_ba, ba[1] / length_ba]
    bc = [unit_vector_ba[0] * distance, unit_vector_ba[1] * distance]
    c = tuple([Fraction(round(a[0] + bc[0])), Fraction(round(a[1] + bc[1]))])
    out = c
    if polygon_area([a, b, c]) != Fraction(0):
        out = perpendicular_line_intersection(a, b, c)
    points.append(out)


def divide_line(line, new_val=None):
    global lines
    global matrix
    global matrix_eq
    global points
    global point_pairs
    a = a2n(line[0])
    b = a2n(line[1])
    if (a, b) not in point_pairs:
        a, b = b, a
        if (a, b) not in point_pairs:
            return None
    new_point = None
    if new_val is None:
        new_point = (
            round((points[a][0] + points[b][0]) / 2),
            round((points[a][1] + points[b][1]) / 2),
        )
        if polygon_area([new_point, points[a], points[b]]) != Fraction(0):
            new_point = perpendicular_line_intersection(points[a], points[b], new_point)
    else:
        new_point = new_val

    point_pairs.pop(point_pairs.index((a, b)))
    point_pairs.append((len(points), a))
    point_pairs.append((len(points), b))
    points.append((Fraction(new_point[0]), Fraction(new_point[1])))


def is_point_on_line(line, point):

    a = points[line[0]]
    b = points[line[1]]
    c = point

    return polygon_area([a, b, c]) == Fraction(0)


def find_line_for_point(point):

    global point_pairs
    output = []
    for i, line in enumerate(point_pairs):
        if is_point_on_line(line, point):
            output.append(i)
    return output


def connect_point(point_ab):
    global lines
    global points
    global point_pairs
    global eq_list
    output = []
    point_a, point_b = point_ab
    point_pairs.append((a2n(point_a), a2n(point_b)))

    inter = find_intersections_2(points, point_pairs)
    
    for p in inter:
      divide_line(line_sort(n2a(p[1][0][0])+n2a(p[1][0][1])), p[0])
      for item in p[1][1:]:
        point_pairs.pop(point_pairs.index(item))
        point_pairs.append((len(points)-1, item[0]))
        point_pairs.append((len(points)-1, item[1]))

def draw_triangle():
    global points
    global point_pairs
    points = [
        (Fraction(400), Fraction(800)),
        (Fraction(800), Fraction(750)),
        (Fraction(600), Fraction(400)),
    ]

    point_pairs = [(0, 1), (1, 2), (2, 0)]


def perpendicular(point, line):
    global points
    global point_pairs
    global eq_list
    global all_angles
    output = perpendicular_line_intersection(
        points[a2n(line[0])], points[a2n(line[1])], points[a2n(point)]
    )
    
    divide_line(line, output)
    perpendicular_angle_list.append(angle_sort(print_angle_3(point+n2a(points.index(output))+line[0])))
    connect_point(n2a(len(points) - 1) + point)


def load_image(file_path):
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk


def run_parallel_function():
    global points
    global point_pairs
    global eq_list
    global print_on
    global matrix
    global matrix_eq
    global all_tri
    global command
    count = 0
    command = command.split("\n")
    command_new = []
    string = None
    while True:
        print_diagram()
        
        if command != []:
            if command[0].split(" ")[0] == "equation":
                command_new.append(command[0])
            string = command.pop(0)
            print_text(">>> ", "green", True, False)
            print_text(string, "blue", True, True)
        else:
            
            command = anglegen13.compute(points, point_pairs, "\n".join(command_new), perpendicular_angle_list)
            for string in command:
                print_text(string, "black", True, True)
            print_text("\nend of program", "green", True, True)
            return
        if string[:13] == "draw triangle":
            draw_triangle()
        elif string.split(" ")[:2] == ["draw", "figure"]:
            points = eval(string.split(" ")[2])
            string_2 = string.split(" ")[3]
            string_2 = string_2[1:-1].split(",")
            for item in string_2:
                point_pairs.append((a2n(item[0]), a2n(item[1])))
        elif string == "draw line":
            points = [(Fraction(400), Fraction(1200)), (Fraction(1200), Fraction(1200))]

            point_pairs = [(0, 1)]
        elif string.split(" ")[0] == "normal" and string.split(" ")[2] == "on":
            p = string.split(" ")[1]
            line = string.split(" ")[3]
            normal_point_fraction(
                points[a2n(line[0])], points[a2n(line[0])], points[a2n(p)]
            )
        elif string == "draw quadrilateral":
            points = [
                (Fraction(400), Fraction(800)),
                (Fraction(800), Fraction(750)),
                (Fraction(600), Fraction(400)),
                (Fraction(400), Fraction(550)),
            ]
            point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif string == "draw right triangle":
            points = [(100, 400), (300, 400), (300, 200)]
            point_pairs = [(0, 1), (1, 2), (2, 0)]
        elif string == "draw quadrilateral":
            points = [(272, 47), (8, 211), (289, 380), (422, 62)]
            point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
        elif string.split(" ")[0] == "perpendicular" and string.split(" ")[2] == "to":
            perpendicular(string.split(" ")[1], string.split(" ")[3])
        
        elif (
            len(string.split(" ")) == 5 and string.split(" ")[0] == "extend"
            and string.split(" ")[2] == "from"
            and string.split(" ")[4] == "for"
        ):
            extend(
                string.split(" ")[1], string.split(" ")[3], int(string.split(" ")[5])
            )
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "from":
            extend(
                string.split(" ")[1], string.split(" ")[3], 200
            )
        elif string.split(" ")[0] == "extend" and string.split(" ")[2] == "to":
            val = find_intersection(
                points[a2n(string.split(" ")[1][0])][0],
                points[a2n(string.split(" ")[1][0])][1],
                points[a2n(string.split(" ")[1][1])][0],
                points[a2n(string.split(" ")[1][1])][1],
                points[a2n(string.split(" ")[3][0])][0],
                points[a2n(string.split(" ")[3][0])][1],
                points[a2n(string.split(" ")[3][1])][0],
                points[a2n(string.split(" ")[3][1])][1],
            )
            divide_line(string.split(" ")[3], val[0])
        elif string.split(" ")[0] == "split":
            divide_line(string.split(" ")[-1])
        elif string.split(" ")[0] == "join":
            connect_point(string.split(" ")[1])


root = tk.Tk()
root.title("geometry Ai")
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
