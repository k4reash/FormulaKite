# Cálculo de la TWD sin Sailmon

Los dispositivos que no registran viento (Vakaros Atlas, Garmin GPX) obtienen la TWD **por maniobras, como Sailmon**: en cada virada o trasluchada limpia, el viento es la **bisectriz** entre el rumbo de antes y el de después.

El cálculo está en `estimate_twd` ([app.py](app.py)). Todos los umbrales son constantes `TWD_*` al principio del archivo.

## 1. TWD de referencia de toda la sesión (`_twd_reference`)

Solo sirve para saber hacia dónde sopla el viento y distinguir viradas de trasluchadas.

- Se cogen los COG con el kite volando a SOG ≥ 12 kts (`TWD_MAN_MIN_KTS`). Hacen falta al menos 300 s (`TWD_REF_MIN_FLYING_S`).
- **Eje del viento** (`_symmetry_axis`): es el eje respecto al que los rumbos son más simétricos, porque las dos amuras se reflejan en la línea del viento. Se usan los armónicos k = 1…8 del COG:
  - Primero se calcula, para cada armónico, la media de cos(k·COG) y de sin(k·COG): son C_k y S_k.
  - Luego se prueba cada eje *a* de 0 a 179°, de grado en grado, con esta puntuación:

  $$\text{score}(a)=\sum_{k=1}^{8}\left[(C_k^2-S_k^2)\cos(2ka)+2C_kS_k\sin(2ka)\right]$$

  - Gana el eje con la puntuación más alta.
- **Sentido**: de las dos opciones (*a* o *a* + 180°), barlovento es el lado donde la SOG media es menor, porque en FK se va más rápido en popa que en ceñida.

## 2. Bisectriz en cada maniobra

Para cada segundo *i* se comparan dos ventanas de 10 s (`TWD_MAN_BEFORE_S`, `TWD_MAN_AFTER_S`), con rumbos medios calculados como medias circulares:

- **Antes**: del segundo i−13 al i−4. Su rumbo medio es θ_b = atan2(media de sin, media de cos).
- **Después**: del segundo i+3 al i+12. Su rumbo medio es θ_a, calculado igual.

Solo cuenta como maniobra válida si se cumple todo esto:

| Condición | Valor | Constante |
|---|---|---|
| Giro \|θ_a − θ_b\| | entre 50° y 130° | `TWD_MAN_TURN_MIN` / `TWD_MAN_TURN_MAX` |
| Dispersión del rumbo √(−2·ln R) en cada ventana (R = longitud del vector medio) | ≤ 20° | `TWD_MAN_MAX_SD` |
| SOG media en las dos ventanas | ≥ 12 kts | `TWD_MAN_MIN_KTS` |
| Huecos en el registro | ninguno grande | — |

Los segundos candidatos se agrupan por maniobra y se toma el de **giro máximo** como centro. La bisectriz es:

$$\text{bis}=\operatorname{atan2}\big(\overline{\sin}_b+\overline{\sin}_a,\ \overline{\cos}_b+\overline{\cos}_a\big)$$

Después se compara la bisectriz con la TWD vigente (al principio, la de referencia del paso 1):

- Si difiere **más de 90°**, es una **trasluchada**: la bisectriz apunta a sotavento, así que se le suman 180°.
- Si aun así difiere **más de 35°** (`TWD_MAN_MAX_DEV`), se descarta. Es una arribada u orzada en baliza, cuya bisectriz queda perpendicular al viento.

## 3. Aplicación en el tiempo

- La nueva TWD se aplica **12 s después** del centro de la maniobra (`TWD_MAN_DELAY_S`), como Sailmon, y se mantiene igual hasta la siguiente maniobra válida.
- **Antes de la primera maniobra no hay viento** (NaN), así que esos segundos quedan como fase "Transición".

## 4. Combinación de todos los regatistas (`borrow_twd`)

Con el origen "Por maniobras (todos los regatistas)":

- Se juntan las estimaciones de todos los tracks.
- En cada segundo se toma la **mediana del seno y del coseno** de la TWD.
- Esas series se interpolan a los tiempos de cada track y la TWD sale de atan2.
- Si no hay ninguna estimación a menos de 60 s (`TWD_BORROW_TOL_S`), ese segundo queda sin viento.

Así todos los regatistas comparten la misma TWD, y TWA, VMG y fases son comparables entre ellos.

## Después, igual que Sailmon (`wind_from_twd`)

$$\text{TWA}=\operatorname{wrap}_{\pm180}(\text{TWD}-\text{COG}),\qquad \text{VMG}=\text{SOG}\cdot\cos(\text{TWA})$$

## Ejemplo: sesión del 05-10 (GPX de Garmin)

| Track | Maniobras válidas | TWD media |
|---|---|---|
| Alex | 42 | ~115° |
| Óscar | 35 | ~125° |
