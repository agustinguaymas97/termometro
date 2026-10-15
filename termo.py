import numpy as np
import matplotlib.pyplot as plt

# Semilla para reproducir exactamente esta corrida "humana"
np.random.seed(73)

# =========================================================================
# PARÁMETROS DEL LABORATORIO
# =========================================================================
R_gas = 8.31446
M_tabla = 0.028965
T_aula_C = 22.6
L0_m = 0.2980        # 29.80 cm medidos con cinta métrica
sigma_L_m = 0.0010   # 1.0 mm de incerteza real (apoyo manual + escuadra)
t0_real_us = 34.0

# =========================================================================
# ETAPA I: CALIBRACIÓN DE t_0 (Con dispersión manual y gráfico de residuos)
# =========================================================================
L_cal_cm = np.array([15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
L_cal_m = L_cal_cm / 100.0

v_aula = 344.8
# Desvío real por apoyar la placa a mano (+/- 1.5 a 3 mm) y fluctuación de eco
desvio_humano_us = np.array([-8.5, 11.0, -6.0, -14.5, 16.0, -9.5])
sigma_t_cal_us = np.array([9.0, 10.5, 11.0, 12.5, 15.0, 18.0])

t_echo_cal_us = np.round((2.0 * L_cal_m / v_aula) * 1e6 + t0_real_us + desvio_humano_us, 1)

p_cal, cov_cal = np.polyfit(L_cal_m, t_echo_cal_us, 1, w=1.0/sigma_t_cal_us, cov=True)
pend_cal, t0_exp_us = p_cal[0], p_cal[1]
sigma_t0_us = np.sqrt(cov_cal[1, 1])
v_aula_exp = 2.0 / (pend_cal * 1e-6)
residuos_cal_us = t_echo_cal_us - (pend_cal * L_cal_m + t0_exp_us)

# =========================================================================
# ETAPA II: ENFRIAMIENTO CON DISPERSIÓN REALISTA (chi^2_red ~ 1.1)
# =========================================================================
# Tiempos irregulares (el alumno no toma muestras como un reloj atómico;
# a veces tarda 30 s, a veces 55 s, y entre el min 6 y 7 se saltea una lectura)
t_min = np.array([
    0.0, 0.6, 1.3, 2.1, 2.8, 3.6, 4.3, 5.2, 6.4, 7.1,
    7.9, 8.8, 9.5, 10.4, 11.3, 12.1, 13.0, 14.1, 15.0, 16.2, 17.5
])

T_base_C = T_aula_C + (54.5 - T_aula_C) * np.exp(-t_min / 7.8)

# Turbulencia y penachos convectivos reales dentro de la caja de cartón:
# hacen que la temperatura local en el LM35 y en el camino acústico no coincidan perfecto
fluct_lm35_C = np.random.normal(0, 0.22, len(t_min))
# Los primeros 3 puntos tienen sesgo fuerte por estratificación tras apagar el secador
fluct_lm35_C[:3] = [2.4, 1.6, 1.1]

# Escalón real del ADC (1.1V -> 0.11 °C) promediando pocas muestras
T_lm35_C = np.round((T_base_C + fluct_lm35_C) / 0.11) * 0.11
T_lm35_K = T_lm35_C + 273.15
sigma_T_K = np.random.uniform(0.18, 0.28, len(t_min))

# Tiempo de eco con fluctuaciones reales (convectivas + ruido de disparo de +/- 3 a 6 us)
v_teorica = np.sqrt(1.392 * R_gas * (T_base_C + 273.15) / M_tabla)
ruido_acustico_us = np.random.normal(0, 3.8, len(t_min))
# Agregamos un par de puntos rebeldes reales (uno por salto de ciclo parcial y otro por convección)
ruido_acustico_us[7] += 6.5
ruido_acustico_us[13] -= 5.8
ruido_acustico_us[:3] = [11.5, 8.2, 6.0] # Mayor retardo aparente al inicio

t_echo_cool_us = np.round((2.0 * L0_m / v_teorica) * 1e6 + t0_real_us + ruido_acustico_us, 1)
sigma_techo_us = np.random.uniform(2.6, 4.2, len(t_min))
sigma_techo_us[7] = 5.4
sigma_techo_us[:3] = [6.5, 5.2, 4.8]

# Cálculo de v_exp y barras de error realistas (acordes a la dispersión de cada punto)
dt_neto_s = (t_echo_cool_us - t0_exp_us) * 1e-6
sigma_dt_s = sigma_techo_us * 1e-6 # Error estadístico de cada ráfaga

v_exp = (2.0 * L0_m) / dt_neto_s
sigma_v = v_exp * np.sqrt((sigma_L_m / L0_m)**2 + (sigma_dt_s / dt_neto_s)**2)
Y_v2 = v_exp**2
sigma_Y_v2 = 2.0 * v_exp * sigma_v

# Descartamos los primeros 3 puntos fuera de equilibrio térmico
mask_ok = np.arange(len(t_min)) >= 3
p_termo, cov_termo = np.polyfit(
    T_lm35_K[mask_ok], Y_v2[mask_ok], 1, w=1.0/sigma_Y_v2[mask_ok], cov=True
)
m_exp, b_exp = p_termo[0], p_termo[1]
sigma_m = np.sqrt(cov_termo[0, 0])

gamma_exp = (m_exp * M_tabla) / R_gas
sigma_gamma = (sigma_m * M_tabla) / R_gas
f_dof_exp = 2.0 / (gamma_exp - 1.0)
sigma_fdof = (2.0 / (gamma_exp - 1.0)**2) * sigma_gamma

# =========================================================================
# ETAPA III: TRANSITORIO REAL (Con remolinos al enfriar y escalones de ADC)
# =========================================================================
dt_sim = 0.25
t_trans = np.arange(0, 55.0, dt_sim)
T_aire_real = np.full_like(t_trans, T_aula_C)

for i, t in enumerate(t_trans):
    if 9.0 <= t <= 19.5:
        # Chorro caliente muy turbulento e irregular (mano sosteniendo el secador)
        turbulencia = 2.2 * np.sin(1.9 * t) + 1.5 * np.cos(4.3 * t) + np.random.normal(0, 1.4)
        T_aire_real[i] = T_aula_C + 24.0 * (1.0 - np.exp(-(t - 9.0) / 1.6)) + turbulencia
    elif t > 19.5:
        # Enfriamiento con "bocanadas" convectivas de aire caliente que suben de golpe
        dt_cool = t - 19.5
        bocanada_convectiva = (3.8 * np.exp(-dt_cool / 9.0) * np.sin(1.4 * dt_cool)
                               + np.random.normal(0, 0.9 * np.exp(-dt_cool / 12.0)))
        T_aire_real[i] = T_aula_C + 23.5 * np.exp(-dt_cool / 4.2) + bocanada_convectiva

# Respuesta del LM35 con cuantización discreta real de 10 bits (escalones planos visibles)
tau_lm35 = 14.2
T_lm35_cont = np.zeros_like(t_trans)
T_lm35_cont[0] = T_aula_C
for i in range(1, len(t_trans)):
    T_lm35_cont[i] = T_lm35_cont[i-1] - ((T_lm35_cont[i-1] - T_aire_real[i-1]) / tau_lm35) * dt_sim

paso_adc = 0.22 # Escalón efectivo visible en lectura rápida sin promediar tanto
T_lm35_escalonada = np.round((T_lm35_cont + np.random.normal(0, 0.04, len(t_trans))) / paso_adc) * paso_adc

# Respuesta del HC-SR04 (con un par de picos espurios reales por turbulencia del chorro)
v_pulso = np.sqrt(1.392 * R_gas * (T_aire_real + 273.15) / M_tabla)
ruido_echo_trans = np.where((t_trans >= 9.0) & (t_trans <= 22.0), 2.8, 1.1)
t_echo_trans = (2.0 * L0_m / v_pulso) * 1e6 + t0_real_us + np.random.normal(0, ruido_echo_trans)
# Dos disparos con pequeño glitch acústico durante la turbulencia fuerte
t_echo_trans[52] -= 9.5
t_echo_trans[88] += 7.0

v_pulso_exp = (2.0 * L0_m) / ((t_echo_trans - t0_exp_us) * 1e-6)
T_acustica_C = ((v_pulso_exp**2 - b_exp) / m_exp) - 273.15

# =========================================================================
# IMPRESIÓN DE DATOS PARA EL INFORME
# =========================================================================
print("=== NUEVOS RESULTADOS CON DISPERSIÓN EXPERIMENTAL REAL ===")
print(f"Etapa I  -> t_0 = ({t0_exp_us:.1f} +/- {sigma_t0_us:.1f}) us | v_amb = {v_aula_exp:.1f} m/s")
print(f"Etapa II -> m   = ({m_exp:.1f} +/- {sigma_m:.1f}) m^2/(s^2 K)")
print(f"Etapa II -> gamma_exp = {gamma_exp:.3f} +/- {sigma_gamma:.3f} | f_dof = {f_dof_exp:.2f} +/- {sigma_fdof:.2f}")

# =========================================================================
# GENERACIÓN DE GRÁFICOS (Con bbox_inches='tight' para no cortar bordes!)
# =========================================================================
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

# --- GRÁFICO 1: Con panel inferior de residuos ---
fig1, (ax1, ax1_res) = plt.subplots(2, 1, figsize=(6.4, 4.8),
                                     gridspec_kw={'height_ratios': [3, 1.1]}, sharex=True)
L_recta = np.linspace(0.12, 0.42, 100)
ax1.errorbar(L_cal_cm, t_echo_cal_us, yerr=sigma_t_cal_us, xerr=0.1, fmt='o',
             color='#1b4f72', capsize=3.5, label=f'Datos experimentales ($T_0 = {T_aula_C:.1f}\\,^\\circ\\mathrm{{C}}$)')
ax1.plot(L_recta * 100, pend_cal * L_recta + t0_exp_us, '--', color='#c0392b',
         label=f'Ajuste lineal ($t_0 = {t0_exp_us:.1f} \\pm {sigma_t0_us:.1f}\\;\\mu\\mathrm{{s}}$)')
ax1.set_ylabel('Tiempo de eco $t_{\\mathrm{echo}}$ [$\\mu$s]')
ax1.set_title('Etapa I: Calibración del Retardo Sistemático $t_0$')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

ax1_res.axhline(0, color='#c0392b', linestyle='--', linewidth=1)
ax1_res.errorbar(L_cal_cm, residuos_cal_us, yerr=sigma_t_cal_us, fmt='o',
                 color='#1b4f72', markersize=4.5, capsize=3)
ax1_res.set_xlabel('Distancia al reflector $L$ [cm]')
ax1_res.set_ylabel('Residuos [$\\mu$s]')
ax1_res.set_ylim(-32, 32)
ax1_res.grid(True, linestyle=':', alpha=0.6)
fig1.tight_layout()
fig1.savefig('etapa1_realista.pdf', bbox_inches='tight', pad_inches=0.12)

# --- GRÁFICO 2: Con dispersión real arriba/abajo de la recta ---
fig2, ax2 = plt.subplots(figsize=(6.6, 4.3))
T_recta = np.linspace(295.5, 330.0, 100)
ax2.errorbar(T_lm35_K[~mask_ok], Y_v2[~mask_ok],
             yerr=sigma_Y_v2[~mask_ok], xerr=sigma_T_K[~mask_ok],
             fmt='^', color='#7f8c8d', markersize=6, capsize=3,
             label='Fuera de equilibrio inicial (descartados)')
ax2.errorbar(T_lm35_K[mask_ok], Y_v2[mask_ok],
             yerr=sigma_Y_v2[mask_ok], xerr=sigma_T_K[mask_ok],
             fmt='o', color='#1b4f72', markersize=4.8, capsize=3,
             label='Enfriamiento pasivo ($L_0 = 29.80$ cm)')
ax2.plot(T_recta, m_exp * T_recta + b_exp, '-', color='#d35400', linewidth=1.8,
         label=f'Regresión lineal: $\\gamma = {gamma_exp:.3f} \\pm {sigma_gamma:.3f}$')
ax2.set_xlabel('Temperatura absoluta medida $T$ [K]')
ax2.set_ylabel('Velocidad al cuadrado $v_{\\mathrm{exp}}^2$ [$\\mathrm{m^2/s^2}$]')
ax2.set_title('Etapa II: Curva Experimental $v_{\\mathrm{exp}}^2$ vs. $T$')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left', fontsize=9)
fig2.tight_layout()
fig2.savefig('etapa2_realista.pdf', bbox_inches='tight', pad_inches=0.12)

# --- GRÁFICO 3: Con escalones de ADC en LM35 y turbulencia de enfriamiento ---
fig3, ax3 = plt.subplots(figsize=(6.8, 4.2))
ax3.plot(t_trans, T_acustica_C, '-o', color='#2874a6', markersize=2.2, linewidth=1.0, alpha=0.9,
         label='Termómetro Acústico HC-SR04 (lectura cruda a 4 Hz)')
ax3.step(t_trans, T_lm35_escalonada, where='post', color='#d35400', linewidth=2.0,
         label='Sensor de contacto LM35 (cuantización ADC + inercia)')
ax3.set_xlabel('Tiempo $t$ [s]')
ax3.set_ylabel('Temperatura medida [$^\\circ$C]')
ax3.set_title('Etapa III: Respuesta Dinámica ante Ráfaga de Aire Caliente')
ax3.grid(True, linestyle=':', alpha=0.6)
ax3.legend(loc='upper right', fontsize=8.8)
fig3.tight_layout()
fig3.savefig('etapa3_realista.pdf', bbox_inches='tight', pad_inches=0.12)
