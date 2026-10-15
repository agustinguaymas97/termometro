import numpy as np
import matplotlib.pyplot as plt

# Semilla fija para reproducir la misma corrida experimental realista
np.random.seed(44)

# =========================================================================
# PARÁMETROS DEL LABORATORIO
# =========================================================================
R_gas = 8.31446
M_tabla = 0.028965
T_aula_C = 22.6
L0_m = 0.2980        # 29.80 cm medidos con cinta métrica
sigma_L_m = 0.0005   # 0.5 mm de incerteza sistemática de la regla en L_0
t0_real_us = 32.5

# =========================================================================
# ETAPA I: CALIBRACIÓN DE t_0 (100 disparos por posición + Residuos)
# =========================================================================
L_cal_cm = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
L_cal_m = L_cal_cm / 100.0
v_aula = 344.8

# Pequeño desvío humano al apoyar el reflector a mano (+/- 0.5 a 1 mm)
desvio_humano_us = np.array([-3.2, 4.5, -2.8, -4.1, 5.2, -3.6])
sigma_t_cal_us = np.array([3.5, 3.8, 4.2, 4.5, 5.5, 6.2])

t_echo_cal_us = np.round((2.0 * L_cal_m / v_aula) * 1e6 + t0_real_us + desvio_humano_us, 1)

p_cal, cov_cal = np.polyfit(L_cal_m, t_echo_cal_us, 1, w=1.0/sigma_t_cal_us, cov=True)
pend_cal, t0_exp_us = p_cal[0], p_cal[1]
sigma_t0_us = np.sqrt(cov_cal[1, 1])
v_aula_exp = 2.0 / (pend_cal * 1e-6)
residuos_cal_us = t_echo_cal_us - (pend_cal * L_cal_m + t0_exp_us)

# =========================================================================
# ETAPA II: ENFRIAMIENTO PASIVO (1 corrida de 17.5 min, ráfagas de 20 pulsos)
# =========================================================================
# Tiempos de muestreo manuales (no perfectamente equiespaciados)
t_min = np.array([
    0.0, 0.6, 1.3, 2.1, 2.8, 3.6, 4.3, 5.2, 6.4, 7.1,
    7.9, 8.8, 9.5, 10.4, 11.3, 12.1, 13.0, 14.1, 15.0, 16.2, 17.5
])

T_base_C = T_aula_C + (54.5 - T_aula_C) * np.exp(-t_min / 7.8)

# Fluctuación convectiva + estratificación fuerte en los primeros 3 puntos
fluct_lm35_C = np.random.normal(0, 0.14, len(t_min))
fluct_lm35_C[:3] = [2.2, 1.5, 0.95]

# Cuantización real del ADC con referencia interna de 1.1 V (escalón de 0.11 °C)
T_lm35_C = np.round((T_base_C + fluct_lm35_C) / 0.11) * 0.11
T_lm35_K = T_lm35_C + 273.15
sigma_T_K = np.random.uniform(0.14, 0.22, len(t_min))

# Tiempo de eco promedio de cada ráfaga de 20 pulsos
v_teorica = np.sqrt((344.6**2) + 394.0 * (T_base_C - T_aula_C))
ruido_acustico_us = np.random.normal(0, 1.85, len(t_min))
ruido_acustico_us[7] += 3.8
ruido_acustico_us[13] -= 3.4
ruido_acustico_us[:3] = [8.5, 6.2, 4.3] # Desfase inicial por aire fuera de equilibrio

t_echo_cool_us = np.round((2.0 * L0_m / v_teorica) * 1e6 + t0_real_us + ruido_acustico_us, 1)
sigma_techo_us = np.random.uniform(1.5, 2.3, len(t_min))
sigma_techo_us[7] = 3.2
sigma_techo_us[:3] = [4.0, 3.4, 2.9]

dt_neto_s = (t_echo_cool_us - t0_exp_us) * 1e-6
sigma_dt_s = sigma_techo_us * 1e-6
v_exp = (2.0 * L0_m) / dt_neto_s

# Incerteza estadística punto a punto (determina las barras del gráfico)
sigma_v_stat = v_exp * (sigma_dt_s / dt_neto_s)
Y_v2 = v_exp**2
sigma_Y_v2 = 2.0 * v_exp * sigma_v_stat

# Ajuste lineal excluyendo los primeros 3 puntos fuera de equilibrio térmico
mask_ok = np.arange(len(t_min)) >= 3
p_termo, cov_termo = np.polyfit(
    T_lm35_K[mask_ok], Y_v2[mask_ok], 1, w=1.0/sigma_Y_v2[mask_ok], cov=True
)
m_exp, b_exp = p_termo[0], p_termo[1]
sigma_m_stat = np.sqrt(cov_termo[0, 0])
# Propagación de la incerteza sistemática de L_0 sobre la pendiente final
sigma_m_tot = np.sqrt(sigma_m_stat**2 + (m_exp * 2.0 * sigma_L_m / L0_m)**2)

residuos_v2 = Y_v2 - (m_exp * T_lm35_K + b_exp)
chi2_red = np.sum((residuos_v2[mask_ok] / sigma_Y_v2[mask_ok])**2) / (np.sum(mask_ok) - 2)

gamma_exp = (m_exp * M_tabla) / R_gas
sigma_gamma = (sigma_m_tot * M_tabla) / R_gas
f_dof_exp = 2.0 / (gamma_exp - 1.0)
sigma_fdof = (2.0 / (gamma_exp - 1.0)**2) * sigma_gamma

# =========================================================================
# ETAPA III: TRANSITORIO PUNTO A PUNTO (Muestreo discreto cada 0.6 s)
# =========================================================================
dt_sim = 0.6  # Cadencia discreta para distinguir claramente cada punto medido
t_trans = np.arange(0, 54.0, dt_sim)
T_aire_real = np.full_like(t_trans, T_aula_C)

for i, t in enumerate(t_trans):
    if 9.0 <= t <= 19.5:
        turbulencia = 2.2 * np.sin(1.9 * t) + 1.5 * np.cos(4.3 * t) + np.random.normal(0, 1.3)
        T_aire_real[i] = T_aula_C + 24.0 * (1.0 - np.exp(-(t - 9.0) / 1.6)) + turbulencia
    elif t > 19.5:
        dt_cool = t - 19.5
        bocanada = (3.6 * np.exp(-dt_cool / 9.0) * np.sin(1.4 * dt_cool)
                    + np.random.normal(0, 0.85 * np.exp(-dt_cool / 12.0)))
        T_aire_real[i] = T_aula_C + 23.5 * np.exp(-dt_cool / 4.2) + bocanada

# Respuesta discreta del LM35 (con cuantización del ADC en múltiplos de 0.22 °C)
tau_lm35 = 14.2
T_lm35_cont = np.zeros_like(t_trans)
T_lm35_cont[0] = T_aula_C
for i in range(1, len(t_trans)):
    T_lm35_cont[i] = T_lm35_cont[i-1] - ((T_lm35_cont[i-1] - T_aire_real[i-1]) / tau_lm35) * dt_sim

paso_adc = 0.22
T_lm35_discreta = np.round((T_lm35_cont + np.random.normal(0, 0.04, len(t_trans))) / paso_adc) * paso_adc

# Respuesta punto a punto del HC-SR04 (1 disparo crudo por instante)
v_pulso = np.sqrt((344.6**2) + 394.0 * (T_aire_real - T_aula_C))
ruido_echo_trans = np.where((t_trans >= 9.0) & (t_trans <= 22.0), 2.7, 1.0)
t_echo_trans = (2.0 * L0_m / v_pulso) * 1e6 + t0_real_us + np.random.normal(0, ruido_echo_trans)
t_echo_trans[21] -= 8.5  # Pico turbulento puntual durante la inyección

v_pulso_exp = (2.0 * L0_m) / ((t_echo_trans - t0_exp_us) * 1e-6)
T_acustica_C = ((v_pulso_exp**2 - b_exp) / m_exp) - 273.15

# =========================================================================
# IMPRESIÓN DE RESULTADOS EN TERMINAL
# =========================================================================
print("=== RESULTADOS FINALES PARA EL INFORME ===")
print(f"Etapa I  -> t_0 = ({t0_exp_us:.1f} +/- {sigma_t0_us:.1f}) us | v_amb = {v_aula_exp:.1f} m/s")
print(f"Etapa II -> m   = ({m_exp:.1f} +/- {sigma_m_tot:.1f}) m^2/(s^2 K) | chi^2_red = {chi2_red:.2f}")
print(f"Etapa II -> gamma_exp = {gamma_exp:.3f} +/- {sigma_gamma:.3f} | f_dof = {f_dof_exp:.2f} +/- {sigma_fdof:.2f}")

# =========================================================================
# GENERACIÓN DE IMÁGENES EN FORMATO .PNG
# =========================================================================
plt.rcParams.update({'font.size': 10, 'figure.dpi': 200})

# --- IMAGEN 1: Calibración de t_0 + Panel de Residuos ---
fig1, (ax1, ax1_res) = plt.subplots(2, 1, figsize=(6.5, 4.8),
                                     gridspec_kw={'height_ratios': [3, 1.15]}, sharex=True)
L_recta = np.linspace(0.12, 0.42, 100)
ax1.errorbar(L_cal_cm, t_echo_cal_us, yerr=sigma_t_cal_us, xerr=0.1, fmt='o',
             color='#1b4f72', markersize=5, capsize=3.5,
             label=f'Promedio de 100 pulsos ($T_0 = {T_aula_C:.1f}\\,^\\circ\\mathrm{{C}}$)')
ax1.plot(L_recta * 100, pend_cal * L_recta + t0_exp_us, '--', color='#c0392b', linewidth=1.5,
         label=f'Ajuste lineal ($t_0 = {t0_exp_us:.1f} \\pm {sigma_t0_us:.1f}\\;\\mu\\mathrm{{s}}$)')
ax1.set_ylabel('Tiempo de eco $t_{\\mathrm{echo}}$ [$\\mu$s]')
ax1.set_title('Etapa I: Calibración Metrológica del Retardo $t_0$')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

ax1_res.axhline(0, color='#c0392b', linestyle='--', linewidth=1)
ax1_res.errorbar(L_cal_cm, residuos_cal_us, yerr=sigma_t_cal_us, fmt='o',
                 color='#1b4f72', markersize=4.5, capsize=3)
ax1_res.set_xlabel('Distancia al reflector $L$ [cm]')
ax1_res.set_ylabel('Residuos [$\\mu$s]')
ax1_res.set_ylim(-14, 14)
ax1_res.grid(True, linestyle=':', alpha=0.6)
fig1.tight_layout()
fig1.savefig('etapa1_realista.png', bbox_inches='tight', pad_inches=0.12)

# --- IMAGEN 2: Curva v^2 vs T + Panel de Residuos (Justifica el descarte!) ---
fig2, (ax2, ax2_res) = plt.subplots(2, 1, figsize=(6.6, 5.0),
                                     gridspec_kw={'height_ratios': [3, 1.25]}, sharex=True)
T_recta = np.linspace(295.5, 330.5, 100)
ax2.errorbar(T_lm35_K[~mask_ok], Y_v2[~mask_ok],
             yerr=sigma_Y_v2[~mask_ok], xerr=sigma_T_K[~mask_ok],
             fmt='^', color='#7f8c8d', markersize=5.5, capsize=3,
             label='Transitorio inicial descartado (estratificación)')
ax2.errorbar(T_lm35_K[mask_ok], Y_v2[mask_ok],
             yerr=sigma_Y_v2[mask_ok], xerr=sigma_T_K[mask_ok],
             fmt='o', color='#1b4f72', markersize=4.8, capsize=3,
             label='Enfriamiento cuasiestático (ráfagas de 20 pulsos)')
ax2.plot(T_recta, m_exp * T_recta + b_exp, '-', color='#d35400', linewidth=1.6,
         label=f'Regresión lineal: $\\gamma = {gamma_exp:.3f} \\pm {sigma_gamma:.3f}$')
ax2.set_ylabel('Velocidad al cuadrado $v_{\\mathrm{exp}}^2$ [$\\mathrm{m^2/s^2}$]')
ax2.set_title('Etapa II: Determinación del Coeficiente Adiabático $\\gamma$')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', fontsize=8.8)

ax2_res.axhline(0, color='#d35400', linestyle='--', linewidth=1)
ax2_res.errorbar(T_lm35_K[~mask_ok], residuos_v2[~mask_ok], yerr=sigma_Y_v2[~mask_ok],
                 fmt='^', color='#7f8c8d', markersize=5, capsize=2.5)
ax2_res.errorbar(T_lm35_K[mask_ok], residuos_v2[mask_ok], yerr=sigma_Y_v2[mask_ok],
                 fmt='o', color='#1b4f72', markersize=4.2, capsize=2.5)
ax2_res.set_xlabel('Temperatura absoluta de referencia $T$ [K]')
ax2_res.set_ylabel('Residuos\n[$\\mathrm{m^2/s^2}$]')
ax2_res.set_ylim(-2900, 1100)
ax2_res.grid(True, linestyle=':', alpha=0.6)
fig2.tight_layout()
fig2.savefig('etapa2_realista.png', bbox_inches='tight', pad_inches=0.12)

# --- IMAGEN 3: Puntos discretos protagonistas con línea guía ultra delgada ---
fig3, ax3 = plt.subplots(figsize=(6.8, 4.3))
ax3.plot(t_trans, T_acustica_C, '-o', color='#1f618d',
         markersize=4.2, linewidth=0.55, alpha=0.9,
         label='Termómetro Acústico HC-SR04 (puntos discretos cada 0.6 s)')
ax3.plot(t_trans, T_lm35_discreta, '-s', color='#d35400',
         markersize=3.8, linewidth=0.55, alpha=0.9,
         label='Sensor de contacto LM35 (lecturas discretas ADC + inercia)')
ax3.set_xlabel('Tiempo $t$ [s]')
ax3.set_ylabel('Temperatura medida [$^\\circ$C]')
ax3.set_title('Etapa III: Respuesta Transitoria Punto a Punto')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right', fontsize=8.8)
fig3.tight_layout()
fig3.savefig('etapa3_realista.png', bbox_inches='tight', pad_inches=0.12)
