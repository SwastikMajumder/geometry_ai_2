excuse arbitrary folder, file and code variable naming. <br>
run the main.py to get this output. this project solves a geometry questions from grade 9 NCERT textbook chapter 7 triangle. NCERT books are relevant in the indian education system. <br>
install the required libraires like ``pip install sympy`` before running the project. <br>
sympy was only used to compute the rref of a matrix. its use was done for a trivial reason <br>
![image](https://github.com/user-attachments/assets/20399e8d-eb28-4719-9910-11194d3d8c71) <br>
```
>>> draw quadrilateral
>>> join AC
>>> join BD
>>> equation angle_val ABC 90
>>> equation line_eq AD CD
>>> equation line_eq CD BC
>>> equation line_eq BC AB
>>> compute
congruent(triangle(ABD),triangle(CBD))
congruent(triangle(BAD),triangle(ADC))
congruent(triangle(ABD),triangle(BAC))
congruent(triangle(BDC),triangle(ACD))
congruent(triangle(BDC),triangle(ACB))
congruent(triangle(ADE),triangle(CDE))
congruent(triangle(AED),triangle(AEB))
congruent(triangle(ACD),triangle(ACB))
congruent(triangle(BEC),triangle(DEC))
congruent(triangle(BEC),triangle(BEA))
congruent(triangle(CED),triangle(AEB))
angle(BED)=180
angle(AEC)=180
angle(ABC)=90
angle(ADC)=90
angle(CAD)=45
angle(ACD)=45
angle(ABD)=45
angle(CBD)=45
angle(ACB)=45
angle(BDC)=45
angle(AED)=90
angle(BEC)=90
angle(ADB)=45
angle(AEB)=90
angle(CED)=90
angle(BCD)=90
angle(BAD)=90
angle(BAC)=45
angle(BED)=angle(AEC)
angle(BAD)=angle(BCD)=angle(AEB)=angle(CED)=angle(AED)=angle(BEC)=angle(ADC)=angle(ABC)
angle(BAC)=angle(ACB)=angle(CBD)=angle(BDC)=angle(ACD)=angle(CAD)=angle(ADB)=angle(ABD)
line(AD)=line(CD)=line(BC)=line(AB)
line(BD)=line(AC)
line(CE)=line(AE)
line(BE)=line(DE)

end of program
```
