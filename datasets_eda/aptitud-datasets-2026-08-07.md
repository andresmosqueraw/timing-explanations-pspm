# Aptitud de datasets para PL-xPsPM — veredicto 2026-08-07

Complementa `comparativa_T21.md`. El screening T2.1 evaluó KPI no saturado, longitud de traza,
vocabulario e intervención plausible. A eso hay que sumar tres requisitos que el propio paper
demostró críticos y que el screening no medía:

- **R3 — n_ΔQ ≥ 30 en test** (Def. 4 de evaluabilidad): los BPI 2020 fallan con 3–8 casos.
- **R6 — comportamiento loggeado estocástico**: si la intervención es casi determinista dado el
  prefijo, la OPE de trayectoria colapsa (ESS→0) y RQ2 es incontestable — el resultado negativo
  del paper en SimBank y BPI 2012.
- **R7 — posición de intervención variable**: si siempre ocurre en el mismo punto relativo del
  caso, "why act now" es trivial. Medida: std de la posición relativa de la 1ª ocurrencia.

## Métricas nuevas (computadas de los parquets del EDA)

| Log | Intervención candidata | % casos | n_test≈ | pos.rel 1ª occ (μ±σ) | reps/caso |
|---|---|---|---|---|---|
| bpic2012 | W_Nabellen incomplete dossiers (la del paper) | 12.6% | 329 | 0.58 ± 0.15 | 15.3 |
| bpic2012 | **W_Nabellen offertes** (alternativa) | 38.3% | 1,003 | 0.41 ± 0.16 | 10.4 |
| bpic2017 | W_Call after offers (la del paper) | 99.5% | 6,272 | 0.41 ± 0.16 | 6.1 |
| bpic2017 | W_Call incomplete files (alternativa) | 47.6% | 3,000 | 0.59 ± 0.13 | 11.2 |
| sepsis | **IV Antibiotics** | 78.4% | 164 | 0.52 ± **0.23** | 1.0 |
| sepsis | IV Liquid | 71.7% | 150 | 0.42 ± 0.23 | 1.0 |
| bpic2014 | Reassignment | 40.1% | 3,736 | 0.28 ± 0.19 | 2.8 |

Outcomes sepsis: Return ER 28.0%, Admission IC 10.5%, Admission NC 76.2%, Release A 63.9%.
Outcome bpic2014 candidato: Reopen 4.5% (casi saturado al revés).

## Veredicto sobre los datasets ACTUALES del paper

| Config | Veredicto | Evidencia (Tabla 1 del paper + EDA) |
|---|---|---|
| SimBank | **Apto** | Sintético diseñado para el problema; banda 32.9; n_ΔQ 69. Su ESS 0.07 es hallazgo, no defecto del log. |
| SimBank-IR3 | **Apto** | Única config w=3; separa los dos niveles. Banda 0.3 (multimodal, conocido). |
| BPI 2012 | **Apto — el mejor log real del estudio** | Intervención no universal (12.6%), posición tardía y variable, 15 puntos de decisión/caso, outcome 17.9% no saturado, banda 6.9, n_ΔQ 192. Falla solo R6 (ESS 0.0004): sirve para explicación, no para el claim de política. |
| BPI 2017 (original) | **No apto como evidencia; apto como caso límite** | Outcome saturado 100%, n_ΔQ 15. Su rol narrativo (el contraste con SLA muestra que la saturación es propiedad del etiquetado) justifica conservarlo. |
| BPI 2017-SLA | **Apto para el margen; débil para riesgo** | Mejor resultado real del paper (86% del margen removido). Banda 1.8 < 3 (no resolvible). Intervención universal (99.5%): solo pregunta "cuándo", nunca "si" — limita Def. 1. |
| BPI 2020 RFP | **No apto** | Trazas media 2.9 (granularidad en el límite), n_ΔQ 8, outcome 93.4%, banda 0.01. |
| BPI 2020 Int-Decl | **No apto** | n_ΔQ 3, outcome 96.0%, banda 0.2. |
| BPI 2020 Travel | **No apto** | n_ΔQ 5, outcome 81.5%, banda 0.4. |

**Conclusión honesta:** de 8 configuraciones, 4 no aportan evidencia positiva evaluable (BPI 2017
original + los 3 BPI 2020). El paper ya lo reporta con candor, pero pagan alquiler en la Tabla 1
sin devolver nada. Recomendación: conservar BPI 2017-original como caso límite (su contraste con
SLA es un hallazgo), y considerar mover los tres BPI 2020 a un apéndice o reducirlos a una fila
resumida "no evaluables por Def. 4". Los BPI 2020 se eligieron por disponibilidad de intervención
(Send Reminder / Request Payment), no por aptitud del KPI — y el KPI resultó saturado en los tres.

## Candidatos NUEVOS

1. **Sepsis — el mejor añadido disponible.**
   - A favor: intervención de libro (IV Antibiotics, respaldada por la literatura PsPM de alarmas),
     una sola ocurrencia por caso (decisión de timing limpia), la mayor varianza posicional de
     todos los candidatos (σ=0.23 → "why now" genuinamente no trivial y mejor pronóstico de
     soporte OPE), outcome Return ER 28% no saturado, trazas mediana 13 (72% ≥10 → masking OK),
     n_ΔQ≈164 ≥ 30 ✓. Dominio médico real: además mitiga las críticas P1/P3 del paper.
   - En contra: 1,050 casos (≈735 train) — riesgo de sobreajuste del transformer d_model=128.
     Mitigable: d_model 32–64, dropout alto, early stopping, y reportar CIs anchos con honestidad.
   - Veredicto: **APTO con reservas de tamaño**. ~~Antes de integrarlo, correr el chequeo R6~~
     **→ Piloto R6 corrido (2026-08-07, `sepsis/pilot_ess.py`): PASA.** Metodología del paper en
     pequeño (MDP con acción leída del sucesor, Q v2 reducido d_model=64, π_b logística
     bag-of-activities, ratios truncados en 20, softmax T=1, 210 casos test):
     - ESS política: **0.140** (gate 0.10) — PASA
     - ESS do-nothing: **0.216** — PASA
     - ESS always-intervene: 0.000 — off-support (esperado: solo 5.8% de decisiones intervienen)

     **Sepsis sería la única configuración del estudio con la política Y do-nothing en soporte a
     la vez con banda potencial** — es decir, la única donde RQ2 (¿la política gana a no hacer
     nada?) sería genuinamente contestable. En el paper, ninguna configuración logró eso con
     resultado positivo. Caveats del piloto: Q pequeño y brevemente entrenado (la política final
     puede divergir más y bajar el ESS); softmax T=1 sobre Q en escala 0/1 queda cerca de
     uniforme, así que el ESS refleja sobre todo la estocasticidad de π_b (misma propiedad
     metodológica que las filas BPI del paper); 210 casos test → CIs anchos.

2. **BPI 2012 con W_Nabellen offertes** (config alternativa sobre log ya integrado).
   - 38.3% de casos, n_test≈1,003 (3× los casos de margen actuales), posición más temprana y
     igual de variable. Coste marginal ~0 (un yaml nuevo). Útil como análisis de robustez:
     ¿las conclusiones dependen de qué intervención se declara?
   - Veredicto: **APTO como config de robustez**, no como log nuevo.

3. **BPI 2014 — descartado pese al tamaño.** 46.6k casos y Reassignment 40% tientan, pero el
   único outcome no trivial (Reopen) tiene base rate 4.5% (saturación inversa), el ciclo está
   casi saturado en 0 (63% bin modal) y las trazas median 7. Confirma el MARGINAL del screening,
   inclinado a NO.

4. **Resto del EDA:** los veredictos NO APTO de `comparativa_T21.md` se sostienen contra los
   criterios adicionales (ninguno rescata R3/R6/R7 lo que falla en R1–R5).

5. **Fuera del EDA:** no hay omisiones obvias en los logs públicos estándar — Helpdesk (Mannhardt)
   es corto y pequeño, BPIC 2016 es clickstream, BPIC 2011 = hospital2011 ya descartado, los PDC
   sintéticos son redundantes con SimBank. La palanca real adicional es el **simulador SimBank**:
   generar configs con comportamiento loggeado más estocástico (mejor soporte OPE por diseño) o
   más anchuras w — es el único entorno donde R6 se controla en vez de heredarse.

## Acción sugerida (si se adopta)

- Añadir `configs/datasets/sepsis.yaml` (intervención IV Antibiotics, outcome Return ER,
  d_model reducido) y correr el pipeline completo con los guards v2.
- Añadir `configs/datasets/bpi2012-offertes.yaml` como robustez.
- Decidir el destino editorial de los BPI 2020 en la reescritura de Fase 4 (apéndice vs. fila
  resumida), no antes de tener los números v2.
