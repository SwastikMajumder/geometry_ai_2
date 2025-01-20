from sympy import Matrix
import itertools
from fractions import Fraction
import copy
from math import gcd
from functools import reduce
def hcf_of_list(numbers):
    return reduce(gcd, numbers)
def lcm(x, y):
    return abs(x * y) // gcd(x, y)
def lcm_of_list(numbers):
    return reduce(lcm, numbers)
def hcf_of_fractions(fractions):
    numerators = [frac.numerator for frac in fractions]
    denominators = [frac.denominator for frac in fractions]
    hcf_numer = hcf_of_list(numerators)
    lcm_denom = lcm_of_list(denominators)
    result = Fraction(hcf_numer, lcm_denom)
    return result

def f2i(given):
    if isinstance(given, int):
        return Fraction(given, 1)
    return given
class EqList:
    def __init__(self):

        self.count = 0
        self.val = {}
        self.pair_eq = []
        self.equation_list = []
        self.equation_list2 = []
        self.var_list = []
        self.newpair = []
    @staticmethod
    def fix_key(a, b):
        for key in a.keys():
            if key not in b.keys():
                b[key] = 0
        for key in b.keys():
            if key not in a.keys():
                a[key] = 0
        return a, b

    @staticmethod
    def valid(equation):
        return not all(key=="=" or equation[key] == Fraction(0) for key in equation.keys())
    
    @staticmethod
    def eq(a, b):
        a, b = EqList.fix_key(a, b)
        return set(a.keys())==set(b.keys()) and all(a[key] == b[key] for key in a.keys())
    
    @staticmethod
    def add(a, b):
        a, b = EqList.fix_key(a, b)
        equation2 = {}
        for key in a.keys():
            equation2[key] = a[key]+b[key]
        return copy.deepcopy(equation2)
    
    @staticmethod
    def sub(a, b):
        a, b = EqList.fix_key(a, b)
        equation2 = {}
        for key in a.keys():
            equation2[key] = a[key]-b[key]
        return copy.deepcopy(equation2)
    
    def include(self, equation):
        
        if len([x for x,y in equation.items() if x != "=" and f2i(y) != Fraction(0)]) == 1:
            tmp = equation["="]/sum([y for x,y in equation.items() if x != "="])
            self.val[[x for x,y in equation.items() if x != "=" and f2i(y) != Fraction(0)][0]] = tmp
        elif sum([y for x,y in equation.items() if x != "="]) == Fraction(0) and len([y for x,y in equation.items() if x != "=" and f2i(y) != Fraction(0)]) == 2 and ("=" not in equation.keys() or f2i(equation["="]) == Fraction(0)):
            tmp = tuple(sorted([x for x,y in equation.items() if x != "=" and f2i(y) != Fraction(0)]))
            if tmp not in self.pair_eq:
                self.pair_eq.append(tmp)
    def total_pair(self):
        for item in self.pair_eq:
            a = None
            b = None
            for i in range(len(self.newpair)):
                if item[0] in self.newpair[i]:
                    a = i
            for i in range(len(self.newpair)):
                if item[1] in self.newpair[i]:
                    b = i
            if a != b and a is not None and b is not None:
                if a > b:
                    a, b = b, a
                newarr = self.newpair[a]+self.newpair[b]
                self.newpair.pop(b)
                self.newpair.pop(a)
                self.newpair.append(newarr)
                
    @staticmethod
    def subs_pair(equation, lhs, rhs):
        if lhs in equation.keys():
            if f2i(equation[lhs]) == Fraction(0):
                return equation
            c = equation[lhs]
            equation[lhs] = 0
            if rhs not in equation.keys():
                equation[rhs] = c
            else:
                equation[rhs] += c
        return equation
    def add_equation2(self, equation, depth):
        equation = copy.deepcopy(equation)
        
        for row in self.newpair:
            if len(row)==1:
                continue
            for j in range(len(row)):
                if j == 0:
                    continue
                equation = EqList.subs_pair(equation, row[j], row[0])
                
        if not EqList.valid(equation) or depth == 0:
            return
        
        if any(EqList.eq(x, equation) for x in self.newlist):
            return
        
        self.newlist.append(equation)
        self.add_equation2(EqList.sub({}, equation), depth-1)
        for item in self.equation_list:
            self.add_equation2(EqList.add(item, equation), depth-1)

        self.include(equation)
        self.total_pair()
        
    def variable(self):
        if "=" in self.var_list:
            self.var_list.remove("=")
        return self.var_list

    def add_equation(self, equation):
        self.newlist = []
        old = self.variable()
        if not any(EqList.eq(x, equation) for x in self.equation_list2):
            self.equation_list2.append(equation)
        else:
            return
        self.equation_list = copy.deepcopy(self.equation_list2)
        self.create_matrix()
        self.matrix = EqList.rref(self.matrix)
        self.create_equation()
        
        self.newpair += [[x] for x in self.var_list if x != "=" and x not in old]
        
        for item in self.equation_list:
            if EqList.valid(item):
                self.include(item)
        
        self.total_pair()
        
        self.add_equation2(equation, 3)
        for item in itertools.combinations(self.val.keys(), 2):
            if self.val[item[0]] == self.val[item[1]]:
                tmp = tuple(sorted(list(item)))
                if tmp not in self.pair_eq:
                    self.pair_eq.append(tmp)
        self.total_pair()
        for item in self.newpair:
            for item2 in itertools.combinations(item, 2):
                tmp = tuple(sorted(list(item2)))
                if tmp not in self.pair_eq:
                    self.pair_eq.append(tmp)
    def create_equation(self):
        self.equation_list = []
        for item in self.matrix:
            tmp = {}
            for index, item2 in enumerate(item):
                tmp[self.var_list[index]] = item2
            self.equation_list.append(tmp)
            
    def create_matrix(self):
        self.var_list = []
        for item in self.equation_list:
            self.var_list += item.keys()
            self.var_list = list(set(self.var_list))
        if "=" in self.var_list:
            self.var_list.remove("=")
        self.var_list.append("=")
        self.matrix = []
        for item in self.equation_list:
            row = [0]*len(self.var_list)
            for key in item.keys():
                row[self.var_list.index(key)] = Fraction(item[key])
            self.matrix.append(row)

    @staticmethod
    def rref(matrix):
        matrix = Matrix(matrix)
        rref_matrix, pivots = matrix.rref()
        return rref_matrix.tolist()

'''
x = EqList()
x.add_equation({"x":1,"y":1,"z":1,"=":180})
x.add_equation({"x":1,"y":-1,"=":0})
x.add_equation({"y":1,"z":-1,"=":0})
print(x.newpair)
'''
