import os
import json
import zipfile
import subprocess
import numpy as np
import pandas as pd


def download_data_if_needed(kaggle_ref, expected_files, data_dir='data'):
    # Crear directorios principales y subdirectorios
    os.makedirs(data_dir, exist_ok=True)
    for subdir in {os.path.dirname(fp) for fp in expected_files}:
        if subdir:
            os.makedirs(os.path.join(data_dir, subdir), exist_ok=True)

    # Verificar si faltan archivos
    existing_files = [os.path.isfile(os.path.join(data_dir, f)) for f in expected_files]
    if all(existing_files):
        print("[INFO] Todos los archivos encontrados en la carpeta data.")
        return

    print("[INFO] Faltan archivos en la carpeta data. Iniciando descarga...")

    # Descargar datos
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", kaggle_ref,
        "-p", data_dir
    ], check=True)

    # Descomprimir archivos zip
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                print(f"[INFO] Descomprimiendo {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                os.remove(zip_path)

    # Reorganizar archivos descargados
    for file_path in expected_files:
        full_path = os.path.join(data_dir, file_path)
        if not os.path.exists(full_path):
            base_name = os.path.basename(file_path)
            candidate_path = os.path.join(data_dir, base_name)

            if os.path.exists(candidate_path):
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                os.rename(candidate_path, full_path)
                print(f"[INFO] Moviendo {candidate_path} a {full_path}")
            else:
                print(f"[WARNING] Archivo esperado {file_path} no encontrado")

    print("[INFO] Proceso completo: Descarga, descompresión y organización finalizadas.")


def process_and_clean_data(folder_path):
    """
    Carga y limpia los archivos en una carpeta especificada según un archivo de configuración.

    Parámetros:
        folder_path (str): Ruta a la carpeta que contiene los archivos CSV y el archivo JSON de configuración.

    Retorna:
        dict: Un diccionario con el nombre del archivo como clave y el DataFrame limpio como valor.
    """

    config_path = os.path.join('config.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    datasets = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            data = pd.read_csv(file_path, low_memory=False)
            base_name = os.path.splitext(file_name)[0]
            columns_to_keep = config.get(base_name, None)

            if columns_to_keep:
                data = data[columns_to_keep]

            if len(data.columns) >= 3:
                data.sort_values(by=list(data.columns[:3]), inplace=True)

            datasets[base_name] = data

    return datasets


def merge_temporal_demographic(datasets):
    keys = sorted(list(datasets.keys()))
    data2 = datasets[keys[2]].drop_duplicates(subset=['nit_enmascarado'])
    data3 = datasets[keys[3]].drop_duplicates(subset=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'])
    data3['fecha_var_rpta_alt'] = (
            pd.to_datetime(data3['fecha_corte'].astype(str), format='%Y%m')
            + pd.DateOffset(months=1)
    ).dt.strftime('%Y%m').astype(int)
    merged = pd.merge(data3, data2, on=['nit_enmascarado'], how='left')
    return merged


def split_data(datasets, merge_data, data_test):
    cutoff = 202311
    data = datasets[sorted(list(datasets.keys()))[0]]
    temporal_train = merge_data[merge_data['fecha_corte'] <= cutoff].copy()
    temporal_test = merge_data[merge_data['fecha_corte'] > cutoff].copy()
    x_train = pd.merge(data, temporal_train, on=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_var_rpta_alt'],
                       how='left')
    x_test = pd.merge(data_test, temporal_test, on=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_var_rpta_alt'],
                      how='left')
    return x_train, x_test


def reorder_columns(train, test):
    """
    Reorganiza las columnas del DataFrame colocando las numéricas primero y las categóricas después.
    """
    numeric_columns_train = train.select_dtypes(include=['number']).columns.tolist()
    categorical_columns_train = train.select_dtypes(exclude=['number']).columns.tolist()
    numeric_columns_test = test.select_dtypes(include=['number']).columns.tolist()
    categorical_columns_test = test.select_dtypes(exclude=['number']).columns.tolist()

    x_train = train[numeric_columns_train + categorical_columns_train].drop(columns=['fecha_corte'])
    x_test = test[numeric_columns_test + categorical_columns_test].drop(columns=['fecha_corte'])
    return x_train, x_test


def impute_missing_values(data):
    """
    Imputa valores faltantes en el DataFrame.
    """
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            # Imputar valores numéricos con la mediana
            data[col] = data[col].fillna(data[col].median())
        else:
            # Imputar valores categóricos con la moda
            data[col] = data[col].fillna(data[col].mode()[0])
    return data


def encode_train_test_together(train, test, target='var_rpta_alt'):
    y_train = train[target].copy()
    train = train.drop(columns=target)
    train['__dataset__'] = 'train'
    test['__dataset__'] = 'test'
    df_concat = pd.concat([train, test], axis=0)

    categorical_columns = df_concat.select_dtypes(include=['object', 'category']).columns.tolist()
    if '__dataset__' in categorical_columns:
        categorical_columns.remove('__dataset__')

    df_encoded = pd.get_dummies(df_concat, columns=categorical_columns, drop_first=True, dtype='int')
    train_encoded = df_encoded[df_encoded['__dataset__'] == 'train'].drop(columns='__dataset__')
    test_encoded = df_encoded[df_encoded['__dataset__'] == 'test'].drop(columns='__dataset__')
    train_encoded[target] = y_train
    return train_encoded, test_encoded


def save_to_parquet(data, name):
    """
    Guarda los DataFrames en formato parquet en una carpeta llamada 'logs'.
    """
    os.makedirs('logs', exist_ok=True)
    data.to_parquet(f'logs/{name}.parquet', index=False)


def main():
    # 1. Descargamos los datos si no existen en la carpeta `data`.
    kaggle_ref = "prueba-analitica-modelo-opciones-de-pago"

    expected_files = [
        "Metadata.xlsx",
        "prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv",
        "prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv",
        "prueba_op_master_customer_data_enmascarado_completa.csv",
        "prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv",
        "test/prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv",
        "test/sample_submission.csv"
    ]

    download_data_if_needed(kaggle_ref, expected_files, data_dir='data')

    # 2. Ejecutamos el procesamiento
    path = 'data/'
    data_clean = process_and_clean_data(path)

    # 3. Cargamos el CSV de test que está en 'data/test'
    data_test_path = os.path.join('data', 'test', 'prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv')
    if not os.path.isfile(data_test_path):
        raise FileNotFoundError(f"El archivo de test no se encuentra en la ruta esperada: {data_test_path}")

    data_to_test = pd.read_csv(data_test_path).drop(columns=['num_oblig_orig_enmascarado'])

    merged_data = merge_temporal_demographic(data_clean)
    temporal_train, temporal_test = split_data(data_clean, merged_data, data_to_test)
    train_, test_ = reorder_columns(temporal_train, temporal_test)
    train, test = impute_missing_values(train_), impute_missing_values(test_)
    data_train, data_test = encode_train_test_together(train, test)
    save_to_parquet(data_train, 'data_train')
    save_to_parquet(data_test, 'data_test')
    print('Done!')


if __name__ == "__main__":
    main()
