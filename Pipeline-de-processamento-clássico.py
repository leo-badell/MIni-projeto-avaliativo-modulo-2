# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Sprint 1 - Configuração e versionamento

# COMMAND ----------

pip install tensorflow

# COMMAND ----------

# DBTITLE 1,Reiniciar Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Cell 2
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# COMMAND ----------

# DBTITLE 1,Carregamento das Imagens
# Caminho base do projeto
base_dir = '/Workspace/Users/leonardobadell@protonmail.com/MIni-projeto-avaliativo-modulo-2'

# Caminhos das pastas com imagens
path_def_front = os.path.join(base_dir, 'def_front')
path_ok_front = os.path.join(base_dir, 'ok_front')

# Listar todos os arquivos de imagem de cada pasta
imagens_def = [os.path.join(path_def_front, f) for f in os.listdir(path_def_front) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]
imagens_ok = [os.path.join(path_ok_front, f) for f in os.listdir(path_ok_front) 
              if f.endswith(('.jpg', '.jpeg', '.png'))]

# Criar DataFrame com Pandas
df_imagens = pd.DataFrame({
    'caminho': imagens_def + imagens_ok,
    'classe': ['def_front'] * len(imagens_def) + ['ok_front'] * len(imagens_ok)
})

# Criar label numérica usando np.where: 1 = defeito, 0 = ok
df_imagens['label'] = np.where(df_imagens['classe'] == 'def_front', 1, 0)

# Exibir informações do dataset
print(f"Total de imagens carregadas: {len(df_imagens)}")
print(f"\nDistribuição de classes:")
print(df_imagens['classe'].value_counts())
print(f"\nPrimeiras 5 linhas do dataset:")
display(df_imagens.head())

# COMMAND ----------

# MAGIC %md
# MAGIC # Sprint 2 - Análise exploratória clássica (OpenCV)

# COMMAND ----------

# MAGIC %md
# MAGIC # Sprint 3 - Destaque de características

# COMMAND ----------

# MAGIC %md
# MAGIC # Sprint 4 - Ingestão de dados e augmentation (Keras)

# COMMAND ----------

# MAGIC %md
# MAGIC # Sprint 5 - A arquitetura CNN e treinamento

# COMMAND ----------

# MAGIC %md
# MAGIC # Sprint 6 - Auditoria e gravação