# Proyecto predicción de probabilidad del riesgo de incumplimiento de pago en clientes de tarjetas de crédito - Team 4 - Despliegue de Soluciones Analíticas

Este proyecto tiene como objetivo desarrollar un modelo de clasificación para predecir la probabilidad del incumplimiento de pagos de clientes de tarjetas de crédito. Utiliza un enfoque de regresión logística y está implementado en Python, con gestión de datos a través de DVC (Data Version Control) y almacenamiento en un bucket de Amazon S3.

## Estructura del proyecto

- `.dvc/`: Carpeta de configuración de DVC para gestionar los datos.
- `Entrega 2/`: Carpeta de scripts que contienen el desarrollo de modelos en Mlflow y dashboard en dash.
- `Entrega Final/`: Carpeta de scripts que contienen el desarrollo de API y Dashboard conectados por Dockerfile.
- `data/`: Carpeta que contiene el archivo de datos `UCI_Credit_Card.csv` rastreado con DVC.
- `.dvcignore`: Archivo que especifica qué archivos y carpetas deben ser ignorados por DVC.
- `Proyecto_entrega_1.ipynb`: Notebook de Jupyter que contiene la exploración de datos y la implementación inicial del modelo.
- `README.md`: Documento de presentación del proyecto, que describe la estructura y propósito del repositorio.


## Instrucciones de Configuración

### 1. Clonar el repositorio

Para clonar este repositorio en la máquina local se debe digitar lo siguiente:

```bash
git clone https://github.com/DanielFlorezT/ProyectoDSA.git
cd ProyectoDSA 
```
### 2. Obtener los datos

Este proyecto usa DVC para gestionar el dataset. Se debe tener acceso al almacenamiento remoto en S3 y ejecutar el siguiente comando para obtener los datos necesarios:

```bash
dvc pull
```

