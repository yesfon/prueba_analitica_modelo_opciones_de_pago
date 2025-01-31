# Prueba Analítica Modelo Opciones de Pago

## Descripción

Este proyecto automatiza la descarga y el procesamiento de datos provenientes de la competencia "Prueba Analítica: Modelo Opciones de Pago" en Kaggle. Utilizando la API de Kaggle, el flujo de trabajo descarga automáticamente los archivos necesarios y realiza la extracción, limpieza, transformación y fusión de datos para generar un dataset listo para el análisis predictivo.

Posteriormente, se aplican técnicas de preprocesamiento (como la imputación de valores faltantes, la codificación de variables categóricas y el escalado de características) y se implementan diversos modelos de Machine Learning, incluyendo SVM, XGBoost, Random Forest y Regresión Logística, para clasificar y evaluar el desempeño mediante métricas como el F1-score.

En terminos generales la idea es hacer una predicción de si un cliente que posee ciertas obligaciones con una entidad bancaria, segun su historial moroso aceptará o no, alguna alternativa de negociación con tal institución. 


## Requisitos y Configuración

### 1. Instalación de Dependencias

Asegúrate de tener Python 3.7 o superior instalado. Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```
## 2. Configuración de Kaggle

Para descargar los datos desde Kaggle, cada usuario debe:

- **Generar su propio archivo `kaggle.json`:**
  - Ingresa a [tu cuenta en Kaggle](https://www.kaggle.com/account).
  - En la sección **API**, haz clic en **Create New API Token** para descargar el archivo.

- **Colocar el archivo en la ubicación adecuada:**

  En Ubuntu, por ejemplo, ejecuta:
  
  ```bash
  mkdir -p ~/.kaggle
  cp /ruta/del/archivo/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```

## 3. Ejecución del Proyecto

Una vez instaladas las dependencias y configurado `kaggle.json`, ejecuta el proyecto con:

```bash
python tu_script.py
```

Asegúrate de que `tu_script.py` sea el archivo principal que orquesta el flujo del proyecto.
