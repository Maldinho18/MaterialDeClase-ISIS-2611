# Plan de envios - Parte 2

Mejor score actual:

- `submission_v2_prior_train_a0.28.csv`: 0.27984
- `submission_v2_prior_train_a0.30.csv`: 0.27984
- `submission_v2_temp0.95_train_a0.25.csv`: 0.27984
- `submission_v2_select_q0.40_train_a0.30.csv`: 0.27984

`submission_v2_select_q0.40_train_a0.30.csv` es duplicado de
`submission_v2_prior_train_a0.30.csv`, segun el log del notebook. En adelante,
no enviar archivos cuyo `duplicate_of` no este vacio en el manifest.

## Siguientes envios recomendados

Enviar pocos y diversos. Orden sugerido:

1. `submission_v2_quota_uniform_a0.35.csv`
2. `submission_v2_prior_train_a0.42.csv`
3. `submission_v2_prior_uniform_a0.38.csv`
4. `submission_v2_smooth0.10_train_a0.30.csv`
5. `submission_v2_temp0.85_train_a0.35.csv`
6. `submission_v2_quota_train_a0.35.csv`

Si alguno sube, explorar alrededor de esa familia. Si todos bajan o empatan, no
seguir mandando variantes cercanas; esperar nuevos logits/probabilidades de otro
modelo o del sweep BETO.

## Regla practica

- No mandar todos los CSV.
- No mandar duplicados indicados por `manifest_postproceso_beto384_v2.csv`.
- Priorizar familias distintas: `prior`, `quota`, `smooth`, `temperature`.
- Mantener seleccionado como mejor actual cualquiera de los `0.27984`.
