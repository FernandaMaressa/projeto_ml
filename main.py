import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading


def rodar_modelos():
    try:
        df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

        X = df.drop("DEATH_EVENT", axis=1)
        y = df["DEATH_EVENT"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def avaliar(modelo, nome):
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)

            return [
                nome,
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred),
            ]

        resultados = []

        for k in [3, 5, 7]:
            resultados.append(avaliar(KNeighborsClassifier(n_neighbors=k), f"KNN (k={k})"))

        svm1 = SVC(kernel="linear", C=1)
        svm2 = SVC(kernel="rbf", C=1)
        svm3 = SVC(kernel="poly", C=1, degree=3)

        resultados.append(avaliar(svm1, "SVM Linear"))
        resultados.append(avaliar(svm2, "SVM RBF"))
        resultados.append(avaliar(svm3, "SVM Polinomial (grau 3)"))

        # MLP
        mlps = [
            MLPClassifier(hidden_layer_sizes=(50,), max_iter=2000, random_state=42),
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
            MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=2000, random_state=42)
        ]
        nomes_mlp = ["MLP (50)", "MLP (100)", "MLP (50-50)"]

        for mlp, nome in zip(mlps, nomes_mlp):
            resultados.append(avaliar(mlp, nome))

        df_res = pd.DataFrame(resultados, columns=["Modelo", "Acurácia", "Precisão", "Recall", "F1 Score"])

        atualizar_tabela(df_res)
        gerar_grafico(df_res)

        messagebox.showinfo("Sucesso", "Modelos executados com sucesso!")

    except Exception as e:
        messagebox.showerror("Erro", str(e))


def gerar_grafico(df):
    plt.figure(figsize=(10, 5))
    plt.bar(df["Modelo"], df["Acurácia"])
    plt.xticks(rotation=45, ha='right')
    plt.title("Comparativo de Acurácia dos Modelos")
    plt.ylabel("Acurácia")
    plt.tight_layout()
    plt.savefig("grafico_acuracia.png", dpi=300)
    plt.close()


def atualizar_tabela(df):

    for linha in tabela.get_children():
        tabela.delete(linha)

    for _, row in df.iterrows():
        tabela.insert("", END, values=list(row))


janela = Tk()
janela.title("Comparação de Modelos de Machine Learning")
janela.geometry("900x500")
janela.config(bg="#F0F4F8")


estilo = ttk.Style()
estilo.theme_use("clam")
estilo.configure("Treeview",
                 background="#e2ecf4",
                 foreground="black",
                 rowheight=28,
                 fieldbackground="#e2ecf4",
                 font=("Segoe UI", 10))
estilo.configure("Treeview.Heading",
                 font=("Segoe UI", 11, "bold"),
                 background="#426aa3",
                 foreground="white")


Label(janela, text="Classificação - Heart Failure Dataset",
      bg="#F0F4F8", font=("Segoe UI", 16, "bold")).pack(pady=10)


frame_tabela = Frame(janela)
frame_tabela.pack(fill="both", expand=True, padx=20, pady=10)

colunas = ["Modelo", "Acurácia", "Precisão", "Recall", "F1 Score"]
tabela = ttk.Treeview(frame_tabela, columns=colunas, show="headings")

for col in colunas:
    tabela.heading(col, text=col)
    tabela.column(col, width=150, anchor='center')

tabela.pack(fill="both", expand=True)


btn = Button(janela, text="Rodar Modelos",
             font=("Segoe UI", 12, "bold"),
             bg="#426aa3", fg="white",
             padx=15, pady=5,
             command=lambda: threading.Thread(target=rodar_modelos).start())

btn.pack(pady=10)

janela.mainloop()
