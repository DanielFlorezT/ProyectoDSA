#!/usr/bin/env python
# coding: utf-8

# **PROYECTO ENTREGA 2 - PREDICCIÓN DE RIESGO DE IMPAGO - GRUPO 4**

# DASHBOARD EN DASH



# -*- coding: utf-8 -*-
import boto3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import boto3
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# SCRIPT PARA DASH 

# se configura el bucket de S3 donde se encuentra la data

bucket_name = 'proyecto-dvcstore-dsa-team4'
file_key = 'files/md5/94/0b416bb13a9b24bb5c9e1589284005'

# Configuración directa de credenciales con libreria boto3 para S3, donde se tuvo que eliminar para poder subir a github por reglas de privacidad.
# Se eliminaron las credenciales para poder subirlo
# Cargar datos desde S3
def cargar_datos():
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(obj['Body'])
    return df

# Cargar y preprocesar los datos
df = cargar_datos()
columnas_relevantes = ['LIMIT_BAL', 'AGE', 'PAY_0', 'SEX', 'EDUCATION', 'MARRIAGE']
valores_por_defecto = {
    'LIMIT_BAL': df['LIMIT_BAL'].mean(),
    'AGE': df['AGE'].mean(),
    'PAY_0': 0,
    'SEX': 2,
    'EDUCATION': 2,
    'MARRIAGE': 2
}

# Normalizar datos y ajustar columnas
scaler = StandardScaler()
df[columnas_relevantes] = scaler.fit_transform(df[columnas_relevantes])
X = df[columnas_relevantes]
y = df['default.payment.next.month']  # Columna target

# Entrenar el modelo - parámetros
modelo = LogisticRegression(max_iter=300, penalty='l2', C=1.0, solver='liblinear')
modelo.fit(X, y)

columnas_modelo = list(X.columns)

# Variables para seguimiento del modelo
precision_modelo = modelo.score(X, y)

# Función para predecir
def predecir(edad=None, limite=None, genero=None, educacion=None, estado=None, pay0=None):
    nueva_data = pd.DataFrame([[
        limite or valores_por_defecto['LIMIT_BAL'],
        edad or valores_por_defecto['AGE'],
        pay0 or valores_por_defecto['PAY_0'],
        genero or valores_por_defecto['SEX'],
        educacion or valores_por_defecto['EDUCATION'],
        estado or valores_por_defecto['MARRIAGE']
    ]], columns=columnas_relevantes)

    # Normalización y encoding
    nueva_data[columnas_relevantes] = scaler.transform(nueva_data[columnas_relevantes])
    probabilidad = modelo.predict_proba(nueva_data)[0][1]
    if probabilidad <= 0.35:
        riesgo = "BAJO"
    elif 0.35 < probabilidad <= 0.65:
        riesgo = "MEDIO"
    else:
        riesgo = "ALTO"
    return probabilidad, riesgo

# Configurar dashboard en Dash
# Configurar dashboard en Dash con estilo
app = Dash(__name__)
app.layout = html.Div(
    style={
        "backgroundColor": "#F7F7F7",  # Fondo general
        "fontFamily": "'Open Sans', sans-serif",
        "padding": "20px",
        "maxWidth": "1200px",
        "margin": "0 auto",
    },
    children=[
        html.H1(
            "Predicción de probabilidad del riesgo de incumplimiento de pago en clientes de tarjetas de crédito",
            style={
                "textAlign": "center",
                "color": "#4E79A7",  # Azul primario
                "paddingBottom": "10px",
            },
        ),
        html.H2(
            "Grupo 4 - Despliegue de Soluciones Analíticas",
            style={"textAlign": "center", "color": "#F28E2C"},
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.P(
                    "En el presente dashboard, puede calcularse el riesgo de que un cliente incumpla "
                    "con sus obligaciones de tarjeta de crédito. Explore diferentes combinaciones "
                    "de variables para identificar patrones y factores que más influyen en el riesgo.",
                    style={"color": "#333333", "fontSize": "16px"},
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Descripción de Variables", style={"color": "#4E79A7"}),
                html.P("Edad: Edad del cliente.", style={"color": "#555555"}),
                html.P("Límite de crédito: Monto máximo aprobado para el cliente.", style={"color": "#555555"}),
                html.P("Género: 1 para masculino, 2 para femenino.", style={"color": "#555555"}),
                html.P(
                    "Educación: Nivel de educación (1=Postgrado, 2=Universitario, etc.).",
                    style={"color": "#555555"},
                ),
                html.P("Estado Civil: 1=Casado, 2=Soltero, etc.", style={"color": "#555555"}),
                html.P(
                    "PAY_0: Estado del pago en el mes de consulta (-1=pagó a tiempo, 1=atraso de 1 mes, ..., 9=atraso de 9 meses o más).",
                    style={"color": "#555555"},
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Panel de Entrada", style={"color": "#4E79A7"}),
                dcc.Input(
                    id="input-edad",
                    type="number",
                    placeholder="Edad",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Input(
                    id="input-limite",
                    type="number",
                    placeholder="Límite de crédito",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Input(
                    id="input-genero",
                    type="number",
                    placeholder="Género",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Input(
                    id="input-educacion",
                    type="number",
                    placeholder="Nivel de Educación",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Input(
                    id="input-estado",
                    type="number",
                    placeholder="Estado Civil",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                dcc.Input(
                    id="input-pay0",
                    type="number",
                    placeholder="Historial de pagos",
                    style={
                        "margin": "10px",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "width": "100%",
                    },
                ),
                html.Button(
                    "Predecir",
                    id="btn-prediccion",
                    style={
                        "backgroundColor": "#4E79A7",
                        "color": "#FFFFFF",
                        "padding": "10px 20px",
                        "border": "none",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                        "marginTop": "20px",
                    },
                ),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Resultado de Predicción", style={"color": "#4E79A7"}),
                html.Div(
                    id="resultado-prediccion",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "color": "#E15759",  # Rojo para destacar
                    },
                ),
                dcc.Graph(id="roc-curve", style={"marginTop": "20px"}),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Factores de Influencia", style={"color": "#4E79A7"}),
                dcc.Graph(id="factores-influencia"),
            ],
        ),
        html.Div(
            style={
                "backgroundColor": "#FFFFFF",
                "padding": "20px",
                "borderRadius": "10px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "marginBottom": "20px",
            },
            children=[
                html.H3("Recomendación", style={"color": "#4E79A7"}),
                html.P(id="recomendacion", style={"color": "#555555", "fontSize": "16px"}),
            ],
        ),
        html.Div(
            style={
                "textAlign": "center",
                "padding": "20px",
                "borderTop": "1px solid #E0E0E0",
                "marginTop": "20px",
                "color": "#4E79A7",
            },
            children=[
                html.H4("Oscar Ardila - Guillermo Ariza - Paola Cifuentes - Daniel Florez Thomas / Grupo 4"),
                html.P("Despliegue de Soluciones Analíticas"),
                html.P("Universidad de los Andes - Maestría en Inteligencia Analítica de Datos"),
            ],
        ),
    ],
)

@app.callback(
    Output("indicadores-modelo", "children"),
    Input("btn-prediccion", "n_clicks")
)
def mostrar_indicadores(n_clicks):
    return f"Precisión del modelo: {precision_modelo:.2f}. AUC-ROC: {auc_roc:.2f}."

@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-prediccion", "n_clicks"),
    State("input-edad", "value"),
    State("input-limite", "value"),
    State("input-genero", "value"),
    State("input-educacion", "value"),
    State("input-estado", "value"),
    State("input-pay0", "value")
)
def actualizar_prediccion(n_clicks, edad, limite, genero, educacion, estado, pay0):
    if n_clicks:
        probabilidad, riesgo = predecir(edad, limite, genero, educacion, estado, pay0)
        return f"Probabilidad de incumplimiento: {probabilidad:.2f}. Riesgo: {riesgo}."

@app.callback(
    Output("roc-curve", "figure"),
    Input("btn-prediccion", "n_clicks")
)
def graficar_roc(n_clicks):
    # Generar la curva ROC
    fpr, tpr, _ = roc_curve(y, modelo.predict_proba(X)[:, 1])
    
    # Crear la figura
    fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode="lines"))
    
    # Actualizar el layout con las métricas
    fig.update_layout(
        title={
            "text": f"Curva ROC<br>Precisión del modelo: {precision_modelo:.2f} | AUC-ROC: {roc_auc_score(y, modelo.predict_proba(X)[:, 1]):.2f}",
            "x": 0.5,  # Centrado
        },
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"  # Estilo más profesional
    )
    return fig

@app.callback(
    Output("factores-influencia", "figure"),
    Input("btn-prediccion", "n_clicks")
)
def mostrar_factores(n_clicks):
    importancia = modelo.coef_[0]
    factores = pd.DataFrame({"Variable": columnas_modelo, "Importancia": importancia}).nlargest(3, "Importancia")
    fig = go.Figure(data=[go.Bar(x=factores["Variable"], y=factores["Importancia"])])
    fig.update_layout(title="Factores de Influencia", xaxis_title="Variables", yaxis_title="Importancia")
    return fig

@app.callback(
    Output("recomendacion", "children"),
    Input("btn-prediccion", "n_clicks")
)
def generar_recomendacion(n_clicks):
    return "Se recomienda establecer alertas tempranas y ajustar políticas para clientes con alto riesgo. Como puede verse los factores más incidentes en el riesgo son el estado de pago en el mes más reciente a la fecha, seguido de la edad y el sexo en una menor proporción."


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8060)