import itertools
import copy
import linear_algebra
import re
from fractions import Fraction

def compute(space, command_given):
    points = space.points
    point_pairs = space.point_pairs
    perpendicular_angle_list = space.perpendicular_angle_list
    list_1 = linear_algebra.EqList()
    list_2 = linear_algebra.EqList()
    eq_list_3 = []
    def n2a(number):
        return chr(number + ord("A"))
    def a2n(letter):
        return ord(letter) - ord("A")
    class TreeNode:
        def __init__(self, name, children=None):
            self.name = name
            self.children = children or []

    def straight_line_2(point_list):
        point_list = [a2n(x) for x in point_list]
        tmp = polygon_area([points[x] for x in point_list])
        return tmp == Fraction(0)

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
                result += "\n" + recursive_str(child, depth + 1)
            return result

        return recursive_str(node)

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

    def find_intersection_line_with_segment(x1, y1, x2, y2, x3, y3, x4, y4):
        def is_within(x1, x2, x):
            return min(x1, x2) <= x <= max(x1, x2)

        # Find intersection point and status
        ans = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        
        # Assume there's no intersection initially
        output = False

        if ans[1] == "intersect":
            ix, iy = ans[0]
            
            # Check if the intersection point is within the segment bounds
            if is_within(x3, x4, ix) and is_within(y3, y4, iy):
                # Check if the intersection is just touching (at an endpoint)
                if (ix, iy) == (x3, y3) or (ix, iy) == (x4, y4):
                    # It's a touch, return False
                    output = False
                else:
                    # Otherwise, it's a real intersection
                    output = True

        return output


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
        lst = list(set(lst))
        new = []
        for item in itertools.permutations(lst):
            if all(
                is_enclosed_angle(points[given_point], item[i], item[i + 1], item[i + 2])
                for i in range(0, len(item) - 2, 1)
            ):
              
                for item2 in item:
                  new.append(n2a(points.index(item2)))
                break

        return new

    def generate_graph():
        graph = dict()
        for i in range(len(points)):
            tmp = surrounding_angle(i)
            graph[n2a(i)] = tmp
        return graph

    def line_sort(line):
        if a2n(line[0]) > a2n(line[1]):
            line = line[1] + line[0]
        return line

    def polygon_area(points):
        n = len(points)
        area = 0
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1] - points[i][1] * points[i + 1][0]
        area += points[-1][0] * points[0][1] - points[-1][1] * points[0][0]
        return abs(area) / 2

    def straight_line(point_list):
        return polygon_area([points[x] for x in point_list]) == 0
    graph = generate_graph()

    def distance(P1, P2):
        x1, y1 = points[P1]
        x2, y2 = points[P2]
        return (x2 - x1)**2 + (y2 - y1)**2  # Squared distance for easier comparison

    def extend_line(P1, P2):
        current_point = P2
        prev_point = P1
        
        # List to store all points tried, starting with P2
        points_tried = [current_point]
        prev_distance = distance(P1, P2)  # Distance from P1 to P2

        while True:
            # Get surrounding points of the current point
            surrounding = graph[n2a(current_point)]
            
            # Find the next point in the same direction that forms a straight line
            next_point = None
            
            for point_label in surrounding:
                point_index = a2n(point_label)
                
                # Skip the point we just came from to avoid backtracking
                if point_index == prev_point:
                    continue
                
                # Check if the new point is along the straight line and further away from P1
                if straight_line([P1, P2, point_index]) and distance(P1, point_index) > prev_distance:
                    next_point = point_index
                    break
            
            if next_point is not None:
                # Update the previous point and distance for the next iteration
                prev_point = current_point
                prev_distance = distance(P1, next_point)
                current_point = next_point
                points_tried.append(current_point)  # Add to the tried points list
            else:
                break  # No valid next point found, stop the loop

        return points_tried, current_point  # Return all points tried and the farthest point





    all_line = []
    for item in point_pairs:
        P1, P2 = item
        forward_points, forward_end = extend_line(P1, P2)
        backward_points, backward_end = extend_line(P2, P1)
        complete_line = backward_points[1:][::-1] + list(item) + forward_points[1:]
        all_line.append([n2a(x) for x in complete_line])

    all_line = [list(x) for i, x in enumerate(all_line) if not any(x == all_line[j] or x[::-1] == all_line[j] for j in range(i))]

    line_counter = []
    for item in itertools.combinations(range(len(points)), 2):
        if item[0] > item[1]:
            item[0], item[1] = item[1], item[0]
        line = n2a(item[0])+n2a(item[1])
        if any(line[0] in x and line[1] in x for x in all_line):
            line_counter.append(line)

    all_angle = []

    angle_counter = []
    for item in itertools.combinations(point_pairs, 2):
        if len(set(list(item[0])+list(item[1]))) == 3:
            b = list(set(item[0]) & set(item[1]))[0]
            a = list(set(item[0]) - set([b]))[0]
            c = list(set(item[1]) - set([b]))[0]
            new_angle = [extend_line(b, a)[1], b, extend_line(b, c)[1]]
            if a > c:
                a, c=  c, a
            if new_angle[0] > new_angle[2]:
                new_angle[0], new_angle[2] = new_angle[2], new_angle[0]
            new_angle = "".join([n2a(x) for x in new_angle])
            angle = n2a(a)+n2a(b)+n2a(c)
            all_angle.append(angle)
            angle_counter.append(new_angle)

    def standard_angle(angle):
        #if not (any(angle[0] in x and angle[1] in x for x in all_line) and any(angle[2] in x and angle[1] in x for x in all_line)):
        #    return None
        a, b, c = a2n(angle[0]), a2n(angle[1]), a2n(angle[2])
        new_angle = [extend_line(b, a)[1], b, extend_line(b, c)[1]]
        if a > c:
            a, c=  c, a
        if new_angle[0] > new_angle[2]:
            new_angle[0], new_angle[2] = new_angle[2], new_angle[0]
        return "".join([n2a(x) for x in new_angle])

    angle_counter  = list(set(angle_counter))

    line_matrix_eq = []
    line_matrix = []

    eq_list = []

    for angle in angle_counter:
        if straight_line([a2n(x) for x in angle]):
            tmp = {}
            tmp[angle] = 1
            tmp["="] = 180
            list_1.add_equation(tmp)

            tmp = {}
            tmp["".join(sorted(angle[0]+angle[1]))] = 1
            tmp["".join(sorted(angle[1]+angle[2]))] = 1
            tmp["".join(sorted(angle[2]+angle[0]))] = -1
            list_2.add_equation(tmp)

    def combine(a, b):

        if a[1] != b[1]:
            return None
        if len(set(a + b)) != 4:
            return None
        r = a[0] + a[2] + b[0] + b[2]
        r = r.replace([x for x in r if r.count(x) == 2][0], "")

        out = list(r[0] + b[1] + r[1])
        if out[0] > out[2]:
            out[0], out[2] = out[2], out[0]

        return "".join(out)
            
    for angle in itertools.permutations(angle_counter, 3):
        if combine(angle[0], angle[1]) == angle[2]:
            
            hhh = [
                a2n(h)
                for h in list(set(angle[0] + angle[1] + angle[2]))
                if list(angle[0] + angle[1] + angle[2]).count(h) == 3
            ][0]
            hh = [
                (a2n(h), hhh)
                for h in list(set(angle[0] + angle[1] + angle[2]))
                if list(angle[0] + angle[1] + angle[2]).count(h) == 2
            ]
            orig = copy.deepcopy(point_pairs)
            point_pairs = hh
            hh = surrounding_angle(hhh)
            point_pairs = copy.deepcopy(orig)

            if hh[1] not in angle[2] and straight_line([a2n(x) for x in angle[2]]):
                tmp = {}
                tmp[angle[0]] = 1
                tmp[angle[1]] = 1
                tmp[angle[2]] = -1
                list_1.add_equation(tmp)


    def angle_sort(angle):
        angle = list(angle)
        if a2n(angle[0]) > a2n(angle[2]):
            angle[0], angle[2] = angle[2], angle[0]
        return "".join(angle)

    for angle in itertools.combinations(angle_counter, 2):
        if (
            angle[0][1] == angle[1][1]
            and straight_line([a2n(x) for x in angle[0]])
            and straight_line([a2n(x) for x in angle[1]])
        ):
            tmp1 = angle_sort(angle[1][0] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[0][0] + angle[1][1] + angle[1][2])
            tmp= {}
            tmp[tmp1] = 1
            tmp[tmp2] = -1
            list_1.add_equation(tmp)
            

            tmp1 = angle_sort(angle[1][2] + angle[0][1] + angle[0][2])
            tmp2 = angle_sort(angle[1][0] + angle[1][1] + angle[0][0])
            tmp= {}
            tmp[tmp1] = 1
            tmp[tmp2] = -1
            list_1.add_equation(tmp)
            

    def is_reflex_vertex(polygon, vertex_index):
        prev_index = (vertex_index - 1) % len(polygon)
        next_index = (vertex_index + 1) % len(polygon)
        modified_polygon = polygon[:vertex_index] + polygon[vertex_index + 1 :]
        original_area = polygon_area(polygon)
        modified_area = polygon_area(modified_polygon)
        if modified_area <= original_area:
            return False
        else:
            return True


    def is_reflex_by_circle(polygon):
        output = []
        for i in range(len(polygon)):
            if is_reflex_vertex(polygon, i):
                output.append(i)
        return output

    all_tri = []

    cycle = []

    def cycle_return(graph, path):
        for item in graph[path[-1]]:
            if item == path[0] and len(path) > 2:
                cycle.append([a2n(x) for x in path])
            elif item not in path:
                cycle_return(graph, path+[item])
    for key in graph.keys():
        cycle_return(graph, [key])

    nn = []
    for item in cycle:
        if set(item) not in [set(x) for x in nn]:
            nn.append(item)

    cycle = nn

    new_cycle = []
    for item in cycle:
        remove_item = []
        for i in range(-2, len(item) - 2, 1):
            if straight_line([item[i], item[i + 1], item[i + 2]]):
                remove_item.append(item[i + 1])
        new_item = item
        for i in range(len(new_item) - 1, -1, -1):
            if new_item[i] in remove_item:
                new_item.pop(i)
        new_cycle.append(new_item)


    for x in new_cycle:
        convex_angle = is_reflex_by_circle([points[y] for y in x])
        
        out = []
        v = None
        for i in range(-2, len(x) - 2, 1):
            angle = [x[i], x[i + 1], x[i + 2]]
            tmp = [[z for z in x][y] for y in convex_angle]

            v = "".join([n2a(y) for y in angle])
            if angle[1] in tmp:
                out.append(
                    "(360-" + standard_angle("".join([n2a(y) for y in angle])) + ")"
                )
            else:
                out.append(standard_angle("".join([n2a(y) for y in angle])))

        if len(x) == 3:
            all_tri.append(v)
        if out == []:
            continue
        copy_out = copy.deepcopy(out)

        out = copy.deepcopy(copy_out)
        for i in range(len(out)):
            out[i] = out[i].replace("(360-", "").replace(")", "")

        subtract = 0
        tmp = {}
        eq_curr = 0
        for i in range(len(out)):
            if "(360-" in copy_out[i]:

                subtract += 180*2
                tmp[out[i]] = -1
                
            else:
                tmp[out[i]] = 1
        
        tmp["="]=180 * (len(x) - 2) - subtract
        list_1.add_equation(tmp)

    all_tri = list(set(all_tri))

    def proof_fx_3(angle1, angle2):
        angle_1 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle1[0]),
                tree_form("d_" + angle1[1]),
                tree_form("d_" + angle1[2]),
            ],
        )
        angle_2 = TreeNode(
            "f_triangle",
            [
                tree_form("d_" + angle2[0]),
                tree_form("d_" + angle2[1]),
                tree_form("d_" + angle2[2]),
            ],
        )
        
        eq = TreeNode("f_congruent", [angle_1, angle_2])
        eq = str_form(eq)
        
        for angle in [angle1 + angle2, angle2 + angle1]:
            
            if sss_rule(*angle) or sas_rule(*angle) or asa_rule(*angle) or rhs_rule(*angle):
                eq_list_3.append(eq)
                do_cpct()
                return eq
        return None

    def line_eq(line1, line2):
        if line1 == line2:
            return True

        return tuple(sorted([line1, line2])) in list_2.pair_eq


    def angle_eq(angle1, angle2):
        if angle1 == angle2:
            return True

        return tuple(sorted([angle1, angle2])) in list_1.pair_eq


    def angle_per(angle):
        if angle not in list_1.val.keys():
            return False
        return list_1.val[angle] == 90

    def sss_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]

        for item in line:
            if item not in line_counter:
                return False

        return (
            line_eq(line[0], line[1])
            and line_eq(line[2], line[3])
            and line_eq(line[4], line[5])
        )


    def sas_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a2 + a3),
            line_sort(b2 + b3),
        ]
        angle = [standard_angle(a1 + a2 + a3), standard_angle(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter:
                return False
        for item in angle:
            if item not in angle_counter:

                return False

        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
        )


    def asa_rule(a1, a2, a3, b1, b2, b3):
        line = [line_sort(a1 + a2), line_sort(b1 + b2)]
        angle = [
            standard_angle(a3 + a1 + a2),
            standard_angle(b3 + b1 + b2),
            standard_angle(a3 + a2 + a1),
            standard_angle(b3 + b2 + b1),
        ]

        for item in line:
            if item not in line_counter:
                return False

        for item in angle:
            if item not in angle_counter:
                return False
            
        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and angle_eq(angle[2], angle[3])
        )
    def string_equation_helper(equation_tree):
        if equation_tree.children == []:
            return equation_tree.name
        s = "("
        if len(equation_tree.children) == 1 or equation_tree.name in [
            "f_if",
            "f_xcongruent",
            "f_congruent",
            "f_triangle",
            "f_xangle",
            "f_xline",
            "f_angle",
            "f_line",
            "f_parallel",
        ]:
            s = equation_tree.name[2:] + s
        sign = {
            "f_if": ",",
            "f_xparallel": ",",
            "f_xcongruent": ",",
            "f_congruent": ",",
            "f_triangle": "?",
            "f_add": "+",
            "f_and": "^",
            "f_dif": "?",
            "f_mul": "*",
            "f_eq": "=",
            "f_sub": "-",
            "f_angle": "?",
            "f_xangle": "?",
            "f_parallel": ",",
            "f_xline": "?",
            "f_exist": "?",
            "f_line": "?",
        }
        for child in equation_tree.children:
            s += string_equation_helper(copy.deepcopy(child)) + sign[equation_tree.name]
        s = s[:-1] + ")"
        return s
    def string_equation(eq):
        eq = eq.replace("d_", "")
        return string_equation_helper(tree_form(eq)).replace("?", "")
    def do_cpct():
        cpct_output = []
        for item in eq_list_3:
            if "congruent" in item:
                m = re.findall(r"[A-Z]{3}", string_equation(item))
                m2 = list(zip(*m))
                for item2 in itertools.permutations(m2):
                    angle1 = "".join([x[0] for x in item2])
                    angle2 = "".join([x[1] for x in item2])
                    tmp = {}
                    tmp[standard_angle(angle1)] = 1
                    tmp[standard_angle(angle2)] = -1
                    list_1.add_equation(tmp)
                
                for item2 in itertools.combinations(m2, 2):
                    line1 = "".join([x[0] for x in item2])
                    line2 = "".join([x[1] for x in item2])
                    line1 = line_sort(line1)
                    line2 = line_sort(line2)
                    if line1 == line2 or tuple(sorted([line1, line2])) in list_2.pair_eq:
                        continue
                    
                    tmp = {}
                    tmp[line1] = 1
                    tmp[line2] = -1
                    list_2.add_equation(tmp)
                    
    def rhs_rule(a1, a2, a3, b1, b2, b3):
        line = [
            line_sort(a1 + a2),
            line_sort(b1 + b2),
            line_sort(a1 + a3),
            line_sort(b1 + b3),
        ]
        angle = [standard_angle(a1 + a2 + a3), standard_angle(b1 + b2 + b3)]

        for item in line:
            if item not in line_counter:
                return False

        for item in angle:
            if item not in angle_counter:
                return False
            
        return (
            line_eq(line[0], line[1])
            and angle_eq(angle[0], angle[1])
            and line_eq(line[2], line[3])
            and angle_per(angle[0])
        )

    def command(string):
        
        eq_type = string.split(" ")[1]
        
        if eq_type == "angle_eq":
            a = standard_angle(string.split(" ")[2])
            b = standard_angle(string.split(" ")[3])
            tmp = {}
            tmp[a] = 1
            tmp[b] = -1
            list_1.add_equation(tmp)
        elif eq_type == "angle_val":
            a = standard_angle(string.split(" ")[2])
            b = int(string.split(" ")[3])
            tmp = {}
            tmp[a] = 1
            tmp["="] = b
            list_1.add_equation(tmp)
        elif eq_type == "line_eq":
            a = line_sort(string.split(" ")[2])
            b = line_sort(string.split(" ")[3])
            tmp = {}
            tmp[a] = 1
            tmp[b] = -1
            list_2.add_equation(tmp)
        elif eq_type == "parallel_line":
            parallel(string)
    def command2(string):
        for item in string.split("\n"):
            command(item)

    
    
    def line_fx(line_input):
        a = line_input[0]
        b = line_input[1]
        return TreeNode("f_line", [tree_form("d_" + a), tree_form("d_" + b)])
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

    def find_all_paths(graph, start_node, end_node, path=[]):
        path = path + [start_node]
        if start_node == end_node:
            return [path]
        if start_node not in graph:
            return []
        paths = []
        for neighbor in graph[start_node]:
            if neighbor not in path:
                new_paths = find_all_paths(graph, neighbor, end_node, path)
                for p in new_paths:
                    paths.append(p)
        return paths

    def is_same_line(line1, line2):
        if line1 == line2:
            return True
        for item in itertools.combinations(range(len(points)), 2):
            for path in find_all_paths(generate_graph(), item[0], item[1]):
                if (
                    straight_line_2(path)
                    and line1[0] in path
                    and line1[1] in path
                    and line2[0] in path
                    and line2[1] in path
                ):
                    return True
        return False
    def proof_fx_2(a, b):
        u, v = a, b
        for item in itertools.combinations(point_pairs, 2):
            if len(set([item[0][0], item[0][1], item[1][0], item[1][1]])) == 4:
                for item2 in itertools.product(item[0], item[1]):
                    if (
                        line_sort(n2a(item2[0]) + n2a(item2[1])) in line_counter
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(u)
                        and line_sort(n2a(item2[0]) + n2a(item2[1])) != line_sort(v)
                    ):
                        c = None
                        d = None
                        if item[0][0] in item2:
                            c = item[0][1]
                        if item[0][1] in item2:
                            c = item[0][0]
                        if item[1][0] in item2:
                            d = item[1][1]
                        if item[1][1] in item2:
                            d = item[1][0]
                        a, b = item2
                        if (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(u))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(v))
                        ) or (
                            is_same_line(line_sort(n2a(a) + n2a(c)), line_sort(v))
                            and is_same_line(line_sort(n2a(b) + n2a(d)), line_sort(u))
                        ):
                            tmp = find_intersection_3(
                                points[c][0],
                                points[c][1],
                                points[d][0],
                                points[d][1],
                                points[a][0],
                                points[a][1],
                                points[b][0],
                                points[b][1],
                            )
                            if tmp[1] == "intersect":
                                tmp = {}
                                tmp[angle_sort(n2a(c) + n2a(a) + n2a(b))] = 1
                                tmp[angle_sort(n2a(d) + n2a(b) + n2a(a))] = -1
                                list_1.add_equation(tmp)
    def parallel(string, dummy=False):
        a = line_sort(string.split(" ")[2])
        b = line_sort(string.split(" ")[3])
        
        a, b = sorted([extend_line2(a), extend_line2(b)])
        
        eq = str_form(
            TreeNode(
                "f_parallel", [line_fx(line_sort(a)), line_fx(line_sort(b))]
            )
        )
        if eq not in eq_list_3:
            eq_list_3.append(eq)
            if not dummy:
                proof_fx_2(a, b)
        
    def two(angle1, angle2):
        list1 = list(itertools.permutations(list(angle1)))
        list2 = list(itertools.permutations(list(angle2)))
        for i in range(len(list1)):
            for j in range(len(list2)):
                out = proof_fx_3("".join(list1[i]), "".join(list2[j]))
                if out is not None:
                    return
    
    def isoceles():
        for item in list_2.pair_eq:
            line1 = item[0]
            line2 = item[1]
            for tri in all_tri:
                if (
                    line1[0] in tri
                    and line2[0] in tri
                    and line1[1] in tri
                    and line2[1] in tri
                ):
                    common = set(line1) & set(line2)
                    common = list(common)[0]
                    a = list(set(line1) - set(common))[0]
                    b = list(set(line2) - set(common))[0]

                    tmp = {}
                    tmp[standard_angle(common + a + b)] = 1
                    tmp[standard_angle(common + b + a)] = -1
                    list_1.add_equation(tmp)
                    break
    
    def extend_line2(var):
        a, b = var
        
        prev = b
        if len(surrounding_angle(a2n(a))) == 2:
            for item in surrounding_angle(a2n(a)):
                
                if item == prev or not straight_line_2(item+prev+a):
                    continue
                prev = a
                a = item
        prev = a
        if len(surrounding_angle(a2n(b))) == 2:
            for item in surrounding_angle(a2n(b)):
                if item == prev or not straight_line_2(item+prev+b):
                    continue
                prev = b
                b = item
        return a+b
    def is_same_direction(var1, var2):
        return extend_line2(var1)==extend_line2(var2)
    def revisoceles():
        for item in list_1.pair_eq:
            for tri in all_tri:
                store = []
                for item2 in itertools.permutations(tri, 3):
                    store.append(standard_angle("".join(item2)))
                
                store = list(set(store))
                for item2 in itertools.combinations(store, 2):
                    if set(item2) == set(item):
                        
                        angle1 = item[0]
                        angle2 = item[1]
                        tmp = {}
                        tmp[line_sort(angle1[0]+angle1[-1])] = 1
                        tmp[line_sort(angle2[0]+angle2[-1])] = -1
                        list_2.add_equation(tmp)
    def revparallel():
        for item in itertools.combinations(list_1.variable(), 2):
            if straight_line_2(item[0]) or straight_line_2(item[1]):
                continue
            lstline = [item[0][1:][::-1], item[0][:-1], item[1][1:][::-1], item[1][:-1]]
            for item2 in itertools.combinations(lstline, 2):
                if is_same_line(*[line_sort(x) for x in item2]) and not is_same_direction(item2[0], item2[1]):
                    if tuple(sorted(list(item))) in list_1.pair_eq:
                        g = list(set(lstline)-set(item2))
                        if not is_same_line(g[0], item2[0]) and not is_same_line(g[1], item2[0]):
                            parallel("equation parallel_line " + " ".join(g))
    for item in perpendicular_angle_list:
        tmp = {}
        tmp[item] = 1
        tmp["="] = Fraction(90)
        list_1.add_equation(tmp)
    command2(command_given)
    
    isoceles()
    revisoceles()
    #revparallel()
    for _ in range(1):
        if len(all_tri) > 1:
            for item in itertools.combinations(all_tri, 2):
                two(item[0], item[1])
                
    output = []
    for item in eq_list_3:
        output.append(string_equation(item))
    for key,val in list_1.val.items():
        output.append("angle("+key+")="+str(val))
    for item in list_1.newpair:
        
        if len(item)>1:
            string = []
            for item2 in item:
                string.append("angle("+item2+")")
            output.append("=".join(string))
    for item in list_2.newpair:
        
        if len(item)>1:
            string = []
            for item2 in item:
                string.append("line("+item2+")")
            output.append("=".join(string))
    return output
