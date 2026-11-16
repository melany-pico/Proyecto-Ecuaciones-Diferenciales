# Neural ODE v0.5 — instrucciones (Windows / Anaconda Prompt)

## Alcance

Esta revisión usa **solo observaciones** de las trayectorias para entrenar. Las derivadas analíticas de `systems.py` se usan únicamente en la evaluación. La normalización de estado se calcula solo con train y se guarda como buffers `state_mean` y `state_std`.

**Advertencia metodológica:** la pérdida de derivadas utiliza diferencias finitas de observaciones con ruido. Su peso `0.1` es un valor exploratorio; no hay garantía de mejora. Para un estudio publicable se debe comparar contra peso `0` con varias semillas y selección de hiperparámetros basada en validación.

## Instalación

1. Haz copia de seguridad de `configs/default.yml`, `scripts/02_train_neural_ode.py`, `scripts/09_evaluate_vector_field.py` y `src/dynadistill/models/neuralode.py`.
2. Copia los archivos del ZIP sobre esas rutas, conservando las carpetas.
3. No sobrescribas `systems.py`, `common.py` ni los datos ya generados.
4. Desde la raíz del proyecto ejecuta:

```bat
conda activate dynadistill
python -m pytest tests -q
python scripts/02_train_neural_ode.py
python scripts/09_evaluate_vector_field.py
```

El entrenamiento guarda `results/checkpoints/lorenz_v05.pt` y `results/metrics/lorenz_training_v05.csv`. El diagnóstico guarda `results/metrics/lorenz_vector_field_v05.csv` y `results/figures/lorenz_vector_field_v05_test.png`.

El checkpoint anterior `lorenz.pt` se conserva. Los scripts de destilación existentes podrían requerir adaptación para leer `lorenz_v05.pt` y transformar derivadas físicas; **no ejecutes la destilación antigua con el nuevo checkpoint sin esa adaptación**.

## Interpretación

La pérdida total se calcula en coordenadas normalizadas:
`L = trajectory_weight * MSE(z_pred,z_obs) + derivative_weight * MSE(dz_pred/dt, finite_difference(z_obs))`.

El evaluador invierte la normalización: `dx/dt = std * f_theta((x-mean)/std)`. Compara con `field(system,x)` en unidades físicas. **No comparar numéricamente** la pérdida normalizada v0.5 con la MSE física v0.2: usar las métricas de campo vectorial del mismo evaluador y mismo split.

## Limitaciones

- Evaluación en estados observados; no mide rollout largo ni exponentes de Lyapunov.
- El muestreo y las diferencias finitas de datos ruidosos pueden sesgar derivadas.
- Validación por trayectorias, pero las condiciones iniciales provienen de la misma distribución.
- El conjunto test debe usarse solo para informe final, no para iterar hiperparámetros.
- La ejecución de 300 épocas en CPU puede tardar bastante; esta entrega ha sido validada en sintaxis, no entrenada de extremo a extremo.
