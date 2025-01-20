import re
import itertools
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk
from fractions import Fraction

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


def surrounding_angle(space, given_point):
    def is_enclosed_angle(curr, h1, h2, h3):
        return find_intersection_line_with_segment(
            curr[0], curr[1], h2[0], h2[1], h1[0], h1[1], h3[0], h3[1]
        )

    lst = []
    for line in space.point_pairs:
        if given_point == line[0]:
            lst.append(space.points[line[1]])
        elif given_point == line[1]:
            lst.append(space.points[line[0]])

    for item in itertools.permutations(lst):
        if all(
            is_enclosed_angle(space.points[given_point], item[i], item[i + 1], item[i + 2])
            for i in range(0, len(item) - 2, 1)
        ):
            lst = list(item)
            break

    tmp = [space.points.index(x) for x in lst]

    return tmp


def n2a(number):
    return chr(number + ord("A"))


def straight_line_2(point_list):
    
    point_list = [a2n(x) for x in point_list]
    tmp = polygon_area([space.points[x] for x in point_list])
    return tmp == Fraction(0)


def straight_line(space, point_list):
    tmp = polygon_area([space.points[x] for x in point_list])

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

def travel_till_end(space, start, step):
    done = False
    step_taken = [step]
    while not done:
        done = True
        for item in surrounding_angle(space, step):
            if (
                straight_line(space, [step, start, item])
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


def sur(space, angle):
    count = 0
    if a2n(angle[0]) in surrounding_angle(space, a2n(angle[1])):
        count += 1
    if a2n(angle[2]) in surrounding_angle(space, a2n(angle[1])):
        count += 1
    return count


def print_angle(space, a, b, c, a_do=True, c_do=True):
    
    if a_do:
        a = travel_till_end(space, b, a)
    else:
        a = travel_till_end(space, b, a)
        a = travel_till_end(space, b, a)
    if c_do:
        c = travel_till_end(space, b, c)
    else:
        c = travel_till_end(space, b, c)
        c = travel_till_end(space, b, c)

    m, n = sorted([a, c])
    return n2a(m) + n2a(b) + n2a(n)


def print_angle_2(space, angle, a_do=True, c_do=True):
    x = angle
    return print_angle(space, a2n(x[0]), a2n(x[1]), a2n(x[2]), a_do, c_do)


def print_angle_3(space, angle):
    lst = [
        print_angle_2(space, angle, True, True),
        print_angle_2(space, angle, True, False),
        print_angle_2(space, angle, False, True),
        print_angle_2(space, angle, False, False),
    ]
    return sorted(lst, key=lambda x: sur(space, x))[0]


def print_angle_4(space, a, b, c):
    return print_angle_3(space, n2a(a) + n2a(b) + n2a(c))


def combine(a, b):

    a = print_angle_3(space, a)
    b = print_angle_3(space, b)
    if a[1] != b[1]:
        return None
    if len(set(a + b)) != 4:
        return None
    r = a[0] + a[2] + b[0] + b[2]
    r = r.replace([x for x in r if r.count(x) == 2][0], "")
    out = print_angle_3(space, r[0] + b[1] + r[1])

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


def extend(space, line, point_start, distance):
 
    b = None
    a = space.points[a2n(point_start)]
    if line[0] == point_start:
        b = space.points[a2n(line[1])]
    else:
        b = space.points[a2n(line[0])]
    ba = [a[0] - b[0], a[1] - b[1]]
    length_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    unit_vector_ba = [ba[0] / length_ba, ba[1] / length_ba]
    bc = [unit_vector_ba[0] * distance, unit_vector_ba[1] * distance]
    c = tuple([Fraction(round(a[0] + bc[0])), Fraction(round(a[1] + bc[1]))])
    out = c
    if polygon_area([a, b, c]) != Fraction(0):
        out = perpendicular_line_intersection(a, b, c)
    space.points.append(out)
    return space


def divide_line(space, line, new_val=None):
    
    a = a2n(line[0])
    b = a2n(line[1])
    if (a, b) not in space.point_pairs:
        a, b = b, a
        if (a, b) not in space.point_pairs:
            return None
    new_point = None
    if new_val is None:
        new_point = (
            round((space.points[a][0] + space.points[b][0]) / 2),
            round((space.points[a][1] + space.points[b][1]) / 2),
        )
        if polygon_area([new_point, space.points[a], space.points[b]]) != Fraction(0):
            new_point = perpendicular_line_intersection(space.points[a], space.points[b], new_point)
    else:
        new_point = new_val

    space.point_pairs.pop(space.point_pairs.index((a, b)))
    space.point_pairs.append((len(space.points), a))
    space.point_pairs.append((len(space.points), b))
    space.points.append((Fraction(new_point[0]), Fraction(new_point[1])))

    return space


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


def connect_point(space, point_ab):
    
    output = []
    point_a, point_b = point_ab
    space.point_pairs.append((a2n(point_a), a2n(point_b)))

    inter = find_intersections_2(space.points, space.point_pairs)
    
    for p in inter:
      space = divide_line(space, line_sort(n2a(p[1][0][0])+n2a(p[1][0][1])), p[0])
      for item in p[1][1:]:
        space.point_pairs.pop(space.point_pairs.index(item))
        space.point_pairs.append((len(space.points)-1, item[0]))
        space.point_pairs.append((len(space.points)-1, item[1]))

    return space
def draw_triangle(space):
    space.points = [
        (Fraction(400), Fraction(800)),
        (Fraction(800), Fraction(750)),
        (Fraction(600), Fraction(400)),
    ]

    space.point_pairs = [(0, 1), (1, 2), (2, 0)]
    return space

def perpendicular(space, point, line):
    output = perpendicular_line_intersection(
        space.points[a2n(line[0])], space.points[a2n(line[1])], space.points[a2n(point)]
    )
    
    space = divide_line(space, line, output)
    space.perpendicular_angle_list.append(angle_sort(print_angle_3(space, point+n2a(space.points.index(output))+line[0])))
    space = connect_point(space, n2a(len(space.points) - 1) + point)
    return space

def draw_quadrilateral(space):
    space.points = [
        (Fraction(400), Fraction(800)),
        (Fraction(800), Fraction(750)),
        (Fraction(600), Fraction(400)),
        (Fraction(400), Fraction(550)),
    ]
    space.point_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return space
