# Comparativa de logs para T2.1 (KPI no saturado)

Criterios: KPI con varianza real (rechazo si >80% de casos en el bin modal del tiempo de ciclo), trazas con longitud suficiente para masking (mediana ≥10 cómodo), vocabulario entrenable, intervención plausible w=2.

| Log | Casos | Acts | Traza med. | Ciclo med./CV | Bin modal | Outcomes binarios | Intervención candidata | Veredicto |
|---|---|---|---|---|---|---|---|---|
| bpic2012 | 13,087 | 24 | 11 (53% ≥10) | 0.8d / CV 1.41 | 51% | approved 17.2%, declined 58.3%, cancelled 21.4% | W_Nabellen offertes (análoga a 2017) | **APTO** |
| bpic2013_closed | 1,487 | 4 | 3 (6% ≥10) | 82.0d / CV 1.33 | 22% | — | — | **NO APTO: trazas demasiado cortas, pocos casos, vocabulario pobre, sin intervención** |
| bpic2013_incidents | 7,554 | 4 | 6 (29% ≥10) | 7.5d / CV 2.37 | 33% | wait_user 0.0% | escalado de línea de soporte (débil) | **NO APTO: trazas justas, vocabulario pobre** |
| bpic2013_open | 819 | 3 | 2 (1% ≥10) | 1.7d / CV 2.91 | 62% | — | — | **NO APTO: trazas demasiado cortas, pocos casos, vocabulario pobre, sin intervención** |
| bpic2014 | 46,616 | 39 | 7 (32% ≥10) | 0.8d / CV 3.39 | 63% | — | reasignación/escalado (débil) | **MARGINAL: trazas justas** |
| bpic2015_1 | 1,199 | 398 | 44 (97% ≥10) | 61.4d / CV 1.27 | 19% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| bpic2015_2 | 832 | 410 | 54 (99% ≥10) | 108.5d / CV 1.05 | 14% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| bpic2015_3 | 1,409 | 383 | 42 (94% ≥10) | 38.5d / CV 1.57 | 16% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| bpic2015_4 | 1,053 | 356 | 44 (98% ≥10) | 96.4d / CV 0.93 | 9% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| bpic2015_5 | 1,156 | 389 | 50 (99% ≥10) | 77.2d / CV 1.10 | 14% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| bpic2017 | 31,509 | 26 | 35 (100% ≥10) | 19.1d / CV 0.60 | 8% | accepted_pending 54.7%, denied 11.9%, cancelled 33.1% | W_Call after offers (ya en pipeline) | **APTO** |
| bpic2018 | 43,809 | 41 | 49 (100% ≥10) | 266.6d / CV 0.47 | 29% | penalty 0.0%, abort 62.9% | — (proceso anual con plazos regulatorios) | **NO APTO: sin intervención** |
| bpic2019 | 251,734 | 42 | 5 (5% ≥10) | 64.0d / CV 2.14 | 6% | clear_invoice 73.0%, block 22.4%, cancel 3.3% | — (compras; sin treatment claro) | **NO APTO: trazas justas, sin intervención** |
| hospital2011 | 1,143 | 624 | 55 (73% ≥10) | 333.0d / CV 0.88 | 17% | — | — | **NO APTO: pocos casos, vocabulario enorme, sin intervención** |
| rtf | 150,370 | 11 | 5 (0% ≥10) | 198.0d / CV 1.02 | 28% | payment 46.4%, credit_collection 39.2%, appeal 3.0% | — (sin punto de intervención del proceso) | **NO APTO: trazas justas, sin intervención** |
| sepsis | 1,050 | 16 | 13 (72% ≥10) | 5.3d / CV 2.13 | 53% | return_er 28.0%, admission_icu 10.5%, admission_nc 76.2%, release_a 63.9% | IV Antibiotics / IV Liquid (literatura PsPM) | **MARGINAL: pocos casos** |

## Notas

- **bpic2012**: Predecesor de BPI 2017, mismo dominio. Outcomes no saturados.
- **bpic2013_closed**: Trazas mediana 3; log pequeño.
- **bpic2013_incidents**: Solo 4 actividades: atribuciones por evento triviales.
- **bpic2013_open**: Trazas mediana 2; log muy pequeño.
- **bpic2014**: Ciclo mediana 0.76d con 63% en el bin modal: casi saturado en cero.
- **bpic2015_1**: KPI de ciclo excelente pero ~1.2k casos, ~400 actividades, sin intervención.
- **bpic2015_2**: Ver municipio 1.
- **bpic2015_3**: Ver municipio 1.
- **bpic2015_4**: Ver municipio 1.
- **bpic2015_5**: Ver municipio 1.
- **bpic2017**: Ya integrado C1–C5. Outcome actual (A_Accepted) saturado al 100% — raíz del problema; corregible con A_Pending (54.7%) o ciclo continuo.
- **bpic2018**: Trazas largas (mediana 49) pero CV 0.47: la duración la fija el calendario anual de la UE, no el manejo del caso; el "penalty" está en atributos de caso, no en actividades.
- **bpic2019**: KPI excelente (modal 5.8%) pero trazas de mediana 5 eventos (4.8% ≥10): mismo problema de masking que RFP.
- **hospital2011**: 624 actividades para 1,143 casos: vocabulario inentrenable; el caso es un historial clínico, no un proceso operativo.
- **rtf**: Trazas de mediana 5 eventos: masking no evaluable.
- **sepsis**: Return ER 28% no saturado, pero solo 1,050 casos.
