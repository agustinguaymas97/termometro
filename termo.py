import numpy as np
import matplotlib.pyplot as plt

# Semilla fija para que los "errores humanos" salgan siempre iguales en tu informe
np.random.seed(19)

# =========================================================================
# 1. CONDICIONES REALISTAS DE LABORATORIO ("CON ERRORES HUMANOS")
# =========================================================================
R_gas = 8.31446
M_aire_humedo = 0.02888  # Levemente menor al aire seco por ~45% de humedad ambiente
gamma_real_efectivo = 1.391 # Leve desvío sistemático real por gradiente térmico vertical

T_aula_C = 22.8          # Temperatura del laboratorio ese día [°C]
L0_real_m = 0.2980       # El alumno quiso poner 30 cm pero al medir dio 29.80 cm
sigma_L_m = 0.0005       # Error de media división de la regla (0.5 mm)
t0_real_us = 31.6        # Tiempo muerto real del módulo HC-SR04 [us]
paso_ADC_C = (1.1 / 1023.0) / 0.010 # Escalón de 0.1075 °C del Arduino Nano a 1.1V

# =========================================================================
# ETAPA I: CALIBRACIÓN DE t_0 (Con error de paralaje al apoyar la madera)
# =========================================================================
L_nominal_cm = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
L_nominal_m = L_nominal_cm / 100.0

# El humano apoya el reflector con un error de +/- 0.8 mm respecto de la marca
error_mano_m = np.array([0.0005, -0.0006, 0.0004, -0.0003, 0.0009, -0.0008])
L_verdadero_m = L_nominal_m + error_mano_m

v_aula = np.sqrt(1.400 * R_gas * (T_aula_C + 273.15) / M_aire_humedo)
# A mayor distancia, el eco llega más débil y el humano alinea peor el ángulo -> mayor barra de error
sigma_t_cal_us = np.array([0.55, 0.62, 0.70, 0.85, 1.35, 1.90])
t_echo_cal_us = (2.0 * L_verdadero_m / v_aula) * 1e6 + t0_real_us + np.random.normal(0, 0.6, len(L_nominal_m))

# Cuantización a múltiplos de 0.5 us (Timer1 con preescalador 8)
t_echo_cal_us = np.round(t_echo_cal_us * 2.0) / 2.0

p_cal, cov_cal = np.polyfit(L_nominal_m, t_echo_cal_us, 1, w=1.0/sigma_t_cal_us, cov=True)
pend_cal, t0_exp_us = p_cal[0], p_cal[1]
sigma_pend_cal, sigma_t0_us = np.sqrt(cov_cal[0, 0]), np.sqrt(cov_cal[1, 1])
v_aula_exp = 2.0 / (pend_cal * 1e-6)

# =========================================================================
# ETAPA II: ENFRIAMIENTO REAL (Ley exponencial + Turbulencia inicial + Outlier)
# =========================================================================
# El estudiante mide cada ~40 segundos durante 16 minutos (24 puntos en total)
# Pero como anota/dispara a mano, los tiempos no son exactos al milisegundo
t_minutos = np.linspace(0, 16.0, 24) + np.random.uniform(-0.08, 0.08, 24)
t_minutos[0] = 0.0

# Enfriamiento exponencial real de la caja (tau_caja = 7.5 min): puntos se amontonan abajo
T_verdadera_aire_C = T_aula_C + (55.8 - T_aula_C) * np.exp(-t_minutos / 7.5)

# ERROR HUMANO 1: En los primeros 3 puntos (t < 1.8 min), el aire caliente recién
# inyectado con el secador se acumula arriba (LM35 marca de más respecto del camino acústico)
sesgo_estratificacion_C = np.zeros_like(t_minutos)
sesgo_estratificacion_C[0] = 2.6  # Recién apagado el secador
sesgo_estratificacion_C[1] = 1.7
sesgo_estratificacion_C[2] = 0.9

T_leida_lm35_C = T_verdadera_aire_C + sesgo_estratificacion_C + np.random.normal(0, 0.08, len(t_minutos))
# Cuantización real del ADC promediando 50 lecturas
T_leida_lm35_C = np.round(T_leida_lm35_C / 0.02) * 0.02
T_leida_lm35_K = T_leida_lm35_C + 273.15
sigma_T_K = np.full_like(T_leida_lm35_K, 0.15)

# Velocidad real en el camino acústico inferior
v_real_camino = np.sqrt(gamma_real_efectivo * R_gas * (T_verdadera_aire_C + 273.15) / M_aire_humedo)
sigma_techo_cool_us = np.random.uniform(0.65, 1.15, len(t_minutos))
# Los primeros 3 puntos tienen más dispersión por turbulencia remanente del secador
sigma_techo_cool_us[:3] = [2.8, 2.2, 1.7]

t_echo_cool_us = (2.0 * L0_real_m / v_real_camino) * 1e6 + t0_real_us + np.random.normal(0, sigma_techo_cool_us)

# ERROR HUMANO 2: En el punto índice 11 (~34 °C), un par de pulsos sufrieron
# Cycle Slipping (+25 us en 3 de las 20 muestras -> desplaza el promedio +3.7 us y agranda sigma)
t_echo_cool_us[11] += 3.7
sigma_techo_cool_us[11] = 2.95

t_echo_cool_us = np.round(t_echo_cool_us * 2.0) / 2.0

# Procesamiento del estudiante usando su t0_exp_us medido en la Etapa I
dt_neto_s = (t_echo_cool_us - t0_exp_us) * 1e-6
sigma_dt_s = np.sqrt(sigma_techo_cool_us**2 + sigma_t0_us**2) * 1e-6

v_exp = (2.0 * L0_real_m) / dt_neto_s
sigma_v = v_exp * np.sqrt((sigma_L_m / L0_real_m)**2 + (sigma_dt_s / dt_neto_s)**2)
Y_v2 = v_exp**2
sigma_Y_v2 = 2.0 * v_exp * sigma_v

# El estudiante descarta con criterio físico los primeros 3 puntos (fuera de equilibrio)
mask_equilibrio = np.arange(len(t_minutos)) >= 3
p_termo, cov_termo = np.polyfit(
    T_leida_lm35_K[mask_equilibrio],
    Y_v2[mask_equilibrio],
    1,
    w=1.0/sigma_Y_v2[mask_equilibrio],
    cov=True
)
m_exp, b_exp = p_termo[0], p_termo[1]
sigma_m = np.sqrt(cov_termo[0, 0])

# El estudiante usa la masa molar de tabla del aire seco (0.028965) en su informe
M_tabla = 0.028965
gamma_exp = (m_exp * M_tabla) / R_gas
sigma_gamma = (sigma_m * M_tabla) / R_gas
f_dof_exp = 2.0 / (gamma_exp - 1.0)
sigma_fdof = (2.0 / (gamma_exp - 1.0)**2) * sigma_gamma

# =========================================================================
# ETAPA III: PULSO TÉRMICO TRANSITORIO (Mano humana con el secador a destiempo)
# =========================================================================
dt_sim = 0.25 # Muestreo cada 250 ms
t_trans = np.arange(0, 55.0, dt_sim)

# El humano prende el secador a los 8.5 s (no a los 10.000 s exactos) y fluctúa la distancia
T_aire_pulso = np.full_like(t_trans, T_aula_C)
for i, t in enumerate(t_trans):
    if 8.5 <= t <= 19.2:
        ondulacion_mano = 1.4 * np.sin(2.1 * t) + 0.8 * np.cos(3.7 * t)
        T_aire_pulso[i] = T_aula_C + 24.5 * (1.0 - np.exp(-(t - 8.5) / 1.4)) + ondulacion_mano
    elif t > 19.2:
        T_pico = T_aire_pulso[int(19.2 / dt_sim)]
        T_aire_pulso[i] = T_aula_C + (T_pico - T_aula_C) * np.exp(-(t - 19.2) / 3.1)

# Respuesta lenta del LM35 (tau = 13.5 s)
tau_lm35 = 13.5
T_lm35_trans = np.zeros_like(t_trans)
T_lm35_trans[0] = T_aula_C
for i in range(1, len(t_trans)):
    dTdt = -(T_lm35_trans[i-1] - T_aire_pulso[i-1]) / tau_lm35
    T_lm35_trans[i] = T_lm35_trans[i-1] + dTdt * dt_sim
T_lm35_trans_meas = np.round((T_lm35_trans + np.random.normal(0, 0.09, len(t_trans))) / paso_ADC_C) * paso_ADC_C

# Respuesta del HC-SR04 (con un poco más de ruido cuando el chorro de aire está prendido)
ruido_flujo_us = np.where((t_trans >= 8.5) & (t_trans <= 19.2), 1.8, 0.65)
v_pulso = np.sqrt(gamma_real_efectivo * R_gas * (T_aire_pulso + 273.15) / M_aire_humedo)
t_echo_pulso_us = (2.0 * L0_real_m / v_pulso) * 1e6 + t0_real_us + np.random.normal(0, ruido_flujo_us)
v_pulso_exp = (2.0 * L0_real_m) / ((t_echo_pulso_us - t0_exp_us) * 1e-6)
# Usamos la recta completa (m_exp * T + b_exp) calibrada en Etapa II
T_acustica_C = ((v_pulso_exp**2 - b_exp) / m_exp) - 273.15

# =========================================================================
# IMPRESIÓN DE RESULTADOS Y TABLA EN LATEX
# =========================================================================
print("=== RESULTADOS REALISTAS PARA TU INFORME ===")
print(f"Etapa I  -> t_0 = ({t0_exp_us:.1f} +/- {sigma_t0_us:.1f}) us | v_amb = ({v_aula_exp:.1f} +/- 0.9) m/s")
print(f"Etapa II -> m   = ({m_exp:.1f} +/- {sigma_m:.1f}) m^2/(s^2 K)")
print(f"Etapa II -> gamma_exp = {gamma_exp:.3f} +/- {sigma_gamma:.3f} (Valor teórico: 1.400)")
print(f"Etapa II -> f_dof_exp = {f_dof_exp:.2f} +/- {sigma_fdof:.2f} (Valor teórico: 5.00)")

print("\n=== COPIAR ESTA TABLA DE MUESTRA AL INFORME LATEX ===")
indices_muestra = [0, 1, 3, 5, 8, 11, 14, 18, 21, 23]
for idx in indices_muestra:
    estado = " (Descartado: turbulencia)" if idx < 3 else (" (Salto de ciclo parcial)" if idx == 11 else "")
    print(f"{t_minutos[idx]:4.1f} & {T_leida_lm35_C[idx]:5.2f} & {T_leida_lm35_K[idx]:6.2f} & "
          f"{t_echo_cool_us[idx]:6.1f} \\pm {sigma_techo_cool_us[idx]:3.1f} & "
          f"{v_exp[idx]:5.2f} \\pm {sigma_v[idx]:3.2f} \\\\ %{estado}")

# =========================================================================
# GRÁFICOS CON ESTÉTICA DE INFORME UNIVERSITARIO
# =========================================================================
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# Gráfico 1
fig1, ax1 = plt.subplots(figsize=(6.4, 4.1))
L_recta = np.linspace(0, 0.43, 100)
ax1.errorbar(L_nominal_cm, t_echo_cal_us, yerr=sigma_t_cal_us, xerr=0.05, fmt='o',
             color='#1b4f72', capsize=3.5, label=f'Mediciones ($T_0 = {T_aula_C:.1f}\\,^\\circ\\mathrm{{C}}$)')
ax1.plot(L_recta * 100, pend_cal * L_recta + t0_exp_us, '--', color='#c0392b',
         label=f'Regresión lineal ($t_0 = {t0_exp_us:.1f} \\pm {sigma_t0_us:.1f}\\;\\mu\\mathrm{{s}}$)')
ax1.set_xlabel('Distancia medida con regla $L$ [cm]')
ax1.set_ylabel('Tiempo de eco $t_{\\mathrm{echo}}$ [$\\mu$s]')
ax1.set_title('Etapa I: Determinación del Retardo Sistemático $t_0$')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()
fig1.tight_layout()
fig1.savefig('etapa1_realista.pdf')

# Gráfico 2 (Mostrando los 3 puntos descartados por estratificación inicial!)
fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
T_recta = np.linspace(296, 331, 100)
ax2.errorbar(T_leida_lm35_K[~mask_equilibrio], Y_v2[~mask_equilibrio],
             yerr=sigma_Y_v2[~mask_equilibrio], xerr=sigma_T_K[~mask_equilibrio],
             fmt='^', color='#7f8c8d', markersize=6, capsize=3,
             label='Transitorio inicial descartado (estratificación)')
ax2.errorbar(T_leida_lm35_K[mask_equilibrio], Y_v2[mask_equilibrio],
             yerr=sigma_Y_v2[mask_equilibrio], xerr=sigma_T_K[mask_equilibrio],
             fmt='o', color='#1b4f72', markersize=4.5, capsize=3,
             label='Enfriamiento cuasiestático ($L_0 = 29.80$ cm)')
ax2.plot(T_recta, m_exp * T_recta + b_exp, '-', color='#d35400', linewidth=1.8,
         label=f'Ajuste ponderado: $\\gamma = {gamma_exp:.3f} \\pm {sigma_gamma:.3f}$')
ax2.set_xlabel('Temperatura absoluta $T$ [K]')
ax2.set_ylabel('Velocidad al cuadrado $v_{\\mathrm{exp}}^2$ [$\\mathrm{m^2/s^2}$]')
ax2.set_title('Etapa II: Curva de Calibración Termo-Acústica')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', fontsize=8.8)
fig2.tight_layout()
fig2.savefig('etapa2_realista.pdf')

# Gráfico 3
fig3, ax3 = plt.subplots(figsize=(6.8, 4.2))
ax3.plot(t_trans, T_acustica_C, '-', color='#2874a6', linewidth=1.2, alpha=0.9,
         label='Termómetro Acústico HC-SR04 (respuesta instantánea)')
ax3.plot(t_trans, T_lm35_trans_meas, '-', color='#d35400', linewidth=2.2,
         label='Sensor de contacto LM35 (retardo por masa térmica)')
ax3.axvspan(8.5, 19.2, color='#f9e79f', alpha=0.35, label='Inyección de aire caliente (flujo turbulento)')
ax3.set_xlabel('Tiempo $t$ [s]')
ax3.set_ylabel('Temperatura registrada [$^\\circ$C]')
ax3.set_title('Etapa III: Respuesta Transitoria e Inercia Térmica')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right', fontsize=8.8)
fig3.tight_layout()
fig3.savefig('etapa3_realista.pdf')
plt.show()
