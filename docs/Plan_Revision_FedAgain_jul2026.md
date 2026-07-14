# Plan de Revisión — FedAgain

**Ms. Ref. No.:** MEDIA-D-26-01276  
**Journal:** Medical Image Analysis  
**Deadline:** 13 Sep 2026

> **Evaluación general:** Ninguno de los dos revisores cuestiona la validez del método ni de los resultados. Los puntos críticos son posicionamiento, justificación teórica y experimentos adicionales — una revisión manejable.

---

## 🔴 Tier 1 — Obligatorio

*Sin estas correcciones el paper no pasa la revisión.*

### 1. Clarificar la novedad vs. FedAgain anterior
**Fuente:** Reviewer 2, punto 1 — el más urgente  
**Tipo:** Solo redacción

Agregar una sección explícita o tabla comparativa *Previous FedAgain vs. This Work*:

| Aspecto | FedAgain anterior | Este trabajo |
|---|---|---|
| Señal de trust | Accuracy-based | Loss-based |
| Análisis de corrupción | No | Corruption-aware |
| Datasets | Limitados | 5 datasets |
| Evaluación de robustez | No | Análisis completo |
| Aplicación clínica | No | Ureteroscopia multi-centro |

---

### 2. Ablación completa del trust score
**Fuente:** Reviewer 1 (punto 5), Reviewer 2 (punto 5)  
**Tipo:** Experimento nuevo

Evaluar por separado:
- Benchmark error only
- Divergence only
- FedAgain combinado

Incluir también la distribución de trust scores por ronda. Sin esto no queda claro si el beneficio viene de alguno de los dos componentes o simplemente de down-weighting clientes inestables.

---

### 3. Análisis de sensibilidad: multiplicativo vs. aditivo
**Fuente:** Reviewer 1 (puntos 1b, 1c)  
**Tipo:** Experimento + análisis

Comparar las tres formulaciones:

- `T_k = 1 / (E_k × D_k)` — formulación actual (multiplicativa)
- `T_k = 1 / (E_k + D_k)` — aditiva
- `T_k = 1 / (α·E_k + β·D_k)` — aditiva ponderada

Mostrar empíricamente que la forma multiplicativa produce mayor supresión de clientes ruidosos y menor varianza. No requiere teorema formal.

---

### 4. Clarificar splits clínicos (patient-level vs. stone-level)
**Fuente:** Reviewer 1, punto 4  
**Tipo:** Solo redacción

Documentar explícitamente:
- ¿El train/test es independiente por paciente?
- ¿Pueden aparecer patches del mismo stone en train y test?
- ¿Cómo se previene el leakage?

Agregar subsección **Clinical Deployment Considerations** y discutir trabajo futuro: video-level inference, agregación temporal, despliegue en tiempo real, validación prospectiva.

---

### 5. Suavizar claims de robustez adversarial
**Fuente:** Reviewer 1 (punto 3), Reviewer 2 (punto 3)  
**Tipo:** Solo redacción

Reemplazar en todo el manuscrito:

> ~~adversarial robustness~~

Por:

> **robustness to corrupted clients and acquisition-related image degradations**

Los experimentos actuales cubren blur, noise y contrast shifts — no label flipping, model poisoning ni Byzantine attacks. Los claims deben reflejar el alcance real.

---

## 🟡 Tier 2 — Muy recomendado

*Aumenta significativamente la probabilidad de aceptación.*

### 6. Agregar baselines: FedBN, Scaffold, MOON
**Fuente:** Reviewer 1 (punto 5), Reviewer 2 (punto 4)  
**Tipo:** Experimento nuevo

Los revisores solicitan: Scaffold, MOON, FedDyn, FedBN, RFLPA, BOBA, RAHFL.

**Prioridad si el tiempo es limitado:**
1. **FedBN** — manejo de heterogeneidad por batch norm
2. **Scaffold** — corrección de drift en non-IID
3. **MOON** — contrastive learning en FL

RFLPA, BOBA y RAHFL son los más relevantes para el claim de robustez (Reviewer 2 los cita explícitamente).

---

### 7. Analizar sesgo contra clientes con distribuciones raras
**Fuente:** Reviewer 2, punto 2  
**Tipo:** Análisis nuevo

Un cliente con clases raras o casos difíciles puede tener alto benchmark error y recibir peso bajo injustamente — aunque su data sea clínicamente valiosa. Analizar si FedAgain suprime clases minoritarias o distribuciones con shift legítimo. Es un punto de equidad del método especialmente relevante en imaging médico.

---

### 8. Reducir y enfocar las Secciones 1 y 3
**Fuente:** Reviewer 1, punto 6  
**Tipo:** Solo redacción

Condensar las partes que leen como tutorial o survey:
- Intro general de AI en medical imaging
- Descripciones detalladas de FedAvg/FedProx
- Discusiones repetidas sobre privacidad y heterogeneidad

El objetivo es un paper metodológico enfocado, no un survey.

---

## 🟢 Tier 3 — Diferenciador

*Eleva el paper a nivel de contribución sólida. Hacer si el tiempo lo permite.*

### 9. Agregar al menos un ataque adversarial real
**Fuente:** Reviewer 1 (punto 3), Reviewer 2 (punto 3)  
**Tipo:** Experimento nuevo

Opciones ordenadas por esfuerzo:
1. **Label Flipping** — más sencillo de implementar
2. **Sign Flipping** — ataque al gradiente
3. **Gaussian Update Attack** — ruido en los pesos

Con un solo experimento de malicious-client el paper gana credibilidad para los claims de robustez. Si no es posible, mantener el claim suavizado del Tier 1, punto 5.

---

### 10. Análisis de convergencia o robustez teórica
**Fuente:** Reviewer 1, punto 2  
**Tipo:** Análisis teórico

No se requiere un teorema completo. Opciones:
- Análisis de convergencia bajo non-IID settings
- Análisis de sensibilidad formal del trust-weight
- Cotas de error bajo distribución de clientes heterogénea

Cualquiera de estos satisface el punto sin necesidad de una prueba exhaustiva.

---

### 11. Reencuadrar el paper: distribution shift + corruption robustness
**Fuente:** Assessment — oportunidad no señalada explícitamente por los revisores  
**Tipo:** Redacción estratégica

El framing actual:
> FL + corruptions + kidney stones

Reposicionarlo como:
> **Trust-aware FL for multi-center distribution shift in endoscopic kidney stone recognition**

Este framing alinea mejor con los intereses actuales de MedIA y refuerza la contribución clínica. Agregar explícitamente la pregunta de investigación:

> *Can trust-aware federated learning improve robustness to inter-hospital distribution shift and image corruption simultaneously?*

---

## Secuencia recomendada de trabajo

1. **Empezar por los puntos de solo redacción** (1, 4, 5, 8) — los más rápidos y desbloquean la narrativa del paper.
2. **Diseñar los experimentos nuevos** (2, 3, 6) para compartir el mismo protocolo y correrlos en paralelo.
3. **El punto más crítico es el #1** — clarificar la novedad vs. FedAgain previo. Sin eso, ningún experimento adicional compensa.
4. **Dejar Tier 3 para el final** como mejoras de oportunidad si el tiempo lo permite.

---

## Resumen de esfuerzo

| # | Acción | Tipo | Tier |
|---|---|---|---|
| 1 | Clarificar novedad vs. FedAgain anterior | Redacción | 🔴 1 |
| 2 | Ablación completa del trust score | Experimento | 🔴 1 |
| 3 | Sensibilidad multiplicativo vs. aditivo | Experimento + análisis | 🔴 1 |
| 4 | Clarificar splits clínicos | Redacción | 🔴 1 |
| 5 | Suavizar claims adversariales | Redacción | 🔴 1 |
| 6 | Baselines: FedBN, Scaffold, MOON | Experimento | 🟡 2 |
| 7 | Sesgo contra clientes con dist. raras | Análisis | 🟡 2 |
| 8 | Condensar Secciones 1 y 3 | Redacción | 🟡 2 |
| 9 | Ataque adversarial real | Experimento | 🟢 3 |
| 10 | Análisis de convergencia | Análisis teórico | 🟢 3 |
| 11 | Reencuadrar framing del paper | Redacción estratégica | 🟢 3 |
