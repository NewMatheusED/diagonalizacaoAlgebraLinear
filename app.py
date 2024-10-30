from flask import Flask, render_template, request
import numpy as np
from sympy import symbols, Matrix, solve, Rational

app = Flask(__name__)

def format_number(num):
    if isinstance(num, complex):
        real = num.real
        imag = num.imag
        if real == int(real):
            real = int(real)
        else:
            real = Rational(real).limit_denominator()
        if imag == int(imag):
            imag = int(imag)
        else:
            imag = Rational(imag).limit_denominator()
        return f"{real} + {imag}i" if imag >= 0 else f"{real} - {-imag}i"
    elif num.is_real:
        if num == int(num):
            return str(int(num))
        else:
            return str(Rational(num).limit_denominator())
    else:
        return str(num).replace("**", "^")

def format_matrix(matrix):
    return "\\begin{pmatrix}" + " \\\\ ".join([" & ".join(map(format_number, row)) for row in matrix.tolist()]) + "\\end{pmatrix}"

def calcular_autovalores_autovetores_passo_a_passo(matriz):
    λ = symbols('λ')
    matriz_sym = Matrix(matriz)
    identidade = Matrix.eye(matriz_sym.shape[0])
    λI = λ * identidade
    A_minus_λI = matriz_sym - λI
    det_A_minus_λI = A_minus_λI.det()
    autovalores = solve(det_A_minus_λI, λ)
    
    passos = {
        "matriz_original": format_matrix(matriz_sym),
        "λI": format_matrix(λI),
        "A_minus_λI": format_matrix(A_minus_λI),
        "det_A_minus_λI": format_number(det_A_minus_λI),
        "autovalores": [format_number(autovalor) for autovalor in autovalores],
        "autovetores": [],
        "P": None,
        "P_inv": None,
        "D": None,
        "diagonalizavel": False
    }
    
    P = []
    for autovalor in autovalores:
        A_minus_λI_val = A_minus_λI.subs(λ, autovalor)
        nullspace = A_minus_λI_val.nullspace()
        if nullspace:
            autovetor = nullspace[0]
            P.append(autovetor)
            passos["autovetores"].append([format_number(val) for val in autovetor])
        else:
            passos["autovetores"].append("Nenhum autovetor encontrado")
    
    if len(P) == matriz_sym.shape[0]:
        P = Matrix.hstack(*P)  # Monta a matriz P com autovetores como colunas
        if P.shape[0] == P.shape[1]:  # Verifica se P é quadrada
            try:
                P_inv = P.inv()
                D = P_inv * matriz_sym * P
                passos["P"] = format_matrix(P)
                passos["P_inv"] = format_matrix(P_inv)
                passos["D"] = format_matrix(D)
                passos["diagonalizavel"] = True
            except Exception as e:
                passos["diagonalizavel"] = False
                passos["erro"] = str(e)
        else:
            passos["diagonalizavel"] = False
            passos["erro"] = "A matriz de autovetores P não é quadrada."
    else:
        passos["diagonalizavel"] = False
        passos["erro"] = "Número insuficiente de autovetores independentes para diagonalizar a matriz."
    
    return passos

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        size = int(request.form.get('matrix-size'))
        matriz = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(float(request.form.get(f'm{i}{j}')))
            matriz.append(row)
        passos = calcular_autovalores_autovetores_passo_a_passo(matriz)
        return render_template('index.html', passos=passos)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)