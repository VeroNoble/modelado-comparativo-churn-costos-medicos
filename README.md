# Modelado comparativo: churn en telecomunicaciones (Telco Customer Churn) y costos médicos de un seguro (Medical Cost Personal)

> 🚧 **En desarrollo.** Trabajo práctico grupal de la Tecnicatura Superior en Ciencia de Datos e IA (materia Modelizado de Minería de Datos). Entrega: noviembre 2026.

Flujo completo de minería de datos aplicado a dos problemas de negocio, comparando técnicas de preprocesamiento, selección de variables, reducción de dimensiones y algoritmos de Machine Learning.

| Problema | Dataset | Variable objetivo | Pregunta de negocio |
| --- | --- | --- | --- |
| **Clasificación binaria** | [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM) — 7.043 clientes | `Churn` (Yes / No) | ¿Qué clientes tienen riesgo de darse de baja? |
| **Regresión** | [Medical Cost Personal](https://www.kaggle.com/datasets/mirichoi0218/insurance) — 1.338 asegurados | `charges` (costo médico anual, USD) | ¿Cuánto va a costarle cada asegurado al seguro? |

## Qué incluye

1. **Análisis exploratorio:** estadística descriptiva, asimetría, distribuciones, correlaciones, matrices de dispersión por clase y boxplots.
2. **Limpieza:** columna numérica guardada como texto, faltantes imputados con criterio de negocio, categorías redundantes, duplicados y atípicos.
3. **Preprocesamiento:** Label, One-Hot y Target Encoding; comparación de MinMax, estandarización, Normalizer, Binarizer, Box-Cox y Yeo-Johnson, siempre ajustados sólo con train y dentro de Pipelines.
4. **Selección de variables y reducción de dimensiones:** RFE con cantidad de variables elegida por validación cruzada (clasificación), importancia con Random Forest (regresión) y PCA sobre las variables del modelo base, con comparación **modelo base vs. selección vs. PCA**.
5. **Modelado:** regresión logística, LDA, k-NN, Naive Bayes, árboles y SVM; regresión lineal, Ridge, LASSO, ElasticNet, k-NN, árboles y SVR, con ajuste manual de hiperparámetros (probando algunos valores por algoritmo) y validación cruzada K-Fold.
6. **Evaluación en test** y conclusiones orientadas al negocio.

## Resultados principales (conjunto de prueba)

**Churn (clasificación):** la **regresión logística** con pesos balanceados logra un **AUC de 0,85** y detecta **8 de cada 10 clientes que se dan de baja** (recall 0,80). Los factores de mayor riesgo son el contrato mes a mes, la poca antigüedad, la fibra óptica y el pago con cheque electrónico. Con la clase desbalanceada (26,5 % de bajas), el accuracy no alcanza: el análisis usa kappa, AUC, F1 y recall.

**Costos médicos (regresión):** la **regresión lineal** explica el **88 % de la variación del costo** (R² 0,877; RMSE ≈ 4.580 USD, contra ≈ 13.100 USD de predecir siempre el promedio). La mayor mejora no vino del algoritmo sino del EDA: detectar que el costo se dispara en los **fumadores con obesidad** y crear esa variable redujo el error un 27 %.

**Selección y PCA:** RFE redujo las variables de 23 a 13 sin perder rendimiento, y la importancia con Random Forest se quedó con 5 de 10. PCA no mejoró en ninguno de los dos problemas y hace perder la interpretación de las variables.

## Cómo correrlo

```bash
pip install -r requirements.txt
jupyter notebook modelado_comparativo_churn_costos_medicos.ipynb
```

Los datasets están en [`data/`](data/).

## Herramientas

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Próximos pasos

- [ ] Completar la consigna de modelado con las tres alternativas (base, selección y PCA)
- [ ] Ajuste sistemático de hiperparámetros (GridSearchCV)
- [ ] Probar modelos de ensamble (Random Forest, Gradient Boosting)
