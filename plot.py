import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.special import j1, jv  # Функции Бесселя первого рода


def plot_intensity_vs_phase(T, R, E0=1.0):
    """
    Строит график зависимости интенсивности от фазы

    Параметры:
    T (float): коэффициент пропускания
    R (float): коэффициент отражения
    E0 (float): амплитуда электрического поля (по умолчанию 1.0)
    """
    # Создаем папку figures, если её нет
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Создаем массив значений фазы от 0 до 2π
    phi = np.linspace(0, 2 * np.pi, 1000)

    # Вычисляем интенсивность
    I = (E0**2 / 2) * (T**2 + R**2 - 2 * T * R * np.cos(phi))

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(phi, I, "b-", linewidth=2)
    plt.xlabel("Фаза φ (рад)", fontsize=12)
    plt.ylabel("Интенсивность I", fontsize=12)
    plt.title(f"Зависимость интенсивности от фазы\nT={T}, R={R}", fontsize=14)
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в папку figures
    plt.savefig(f"figures/intensity_vs_phase_T{T}_R{R}.png")
    plt.show()


def plot_bessel_intensity(T, R, phi0, Phi_max=10, I_avg=1.0):
    """
    Строит график зависимости I от Phi для выражения 4*I_avg*T*R*sin(phi0)*J1(Phi)

    Параметры:
    I_avg (float): средняя интенсивность
    T (float): коэффициент пропускания
    R (float): коэффициент отражения
    phi0 (float): начальная фаза
    Phi_max (float): максимальное значение Phi для графика
    """
    # Создаем папку figures, если её нет
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Создаем массив значений Phi
    Phi = np.linspace(0, Phi_max, 1000)

    # Вычисляем интенсивность
    I = 4 * I_avg * T * R * np.sin(phi0) * j1(Phi)

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(Phi, I, "b-", linewidth=2)
    plt.xlabel("Φ", fontsize=12)
    plt.ylabel("Интенсивность I", fontsize=12)
    plt.title(
        f"Зависимость интенсивности от Φ\nI_avg={I_avg}, T={T}, R={R}, φ₀={phi0:.2f}",
        fontsize=14,
    )
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в папку figures
    plt.savefig(f"figures/bessel_intensity_I{I_avg}_T{T}_R{R}_phi{phi0:.2f}.png")
    plt.show()


def plot_bessel2_intensity(T, R, phi0, Omega_t=0, Phi_max=10, I_avg=1.0):
    """
    Строит график зависимости I от Phi для выражения 4*I_avg*T*R*cos(phi0)*J2(Phi)*cos(2*Omega*t)
    
    Параметры:
    T (float): коэффициент пропускания
    R (float): коэффициент отражения
    phi0 (float): начальная фаза
    Omega_t (float): произведение частоты на время (по умолчанию 0)
    Phi_max (float): максимальное значение Phi для графика
    I_avg (float): средняя интенсивность
    """
    # Создаем папку figures, если её нет
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Создаем массив значений Phi
    Phi = np.linspace(0, Phi_max, 1000)

    # Вычисляем интенсивность, используя jv(2, Phi) вместо j2(Phi)
    I = 4 * I_avg * T * R * np.cos(phi0) * jv(2, Phi) * np.cos(2 * Omega_t)

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(Phi, I, "b-", linewidth=2)
    plt.xlabel("Φ", fontsize=12)
    plt.ylabel("Интенсивность I", fontsize=12)
    plt.title(
        f"Зависимость интенсивности от Φ\nI_avg={I_avg}, T={T}, R={R}, φ₀={phi0:.2f}, Ωt={Omega_t:.2f}",
        fontsize=14,
    )
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в папку figures
    plt.savefig(f"figures/bessel2_intensity_I{I_avg}_T{T}_R{R}_phi{phi0:.2f}_Omega{Omega_t:.2f}.png")
    plt.show()


def plot_first_harmonic_intensity(T0, T, R0, R, phi0, W, Phi_max=10, I_avg=1.0):
    """
    Строит график зависимости I₁ от Phi для первой гармоники
    
    Параметры:
    T0 (float): коэффициент пропускания первого плеча
    T (float): коэффициент пропускания второго плеча
    R0 (float): коэффициент отражения первого плеча
    R (float): коэффициент отражения второго плеча
    phi0 (float): начальная фаза
    W (float): параметр W
    Phi_max (float): максимальное значение Phi для графика
    I_avg (float): средняя интенсивность
    """
    # Создаем папку figures, если её нет
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Создаем массив значений Phi
    Phi = np.linspace(0, Phi_max, 1000)

    # Вычисляем A и B
    A = (T0 * T + R0 * R) * j1(W * Phi)
    B = T0 * R0 * j1(Phi)

    # Вычисляем C1
    C1 = 2 * I_avg * np.sqrt(A**2 + B**2 - 2 * A * B * np.cos((W - 1) * phi0))

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(Phi, C1, "b-", linewidth=2)
    plt.xlabel("Φ", fontsize=12)
    plt.ylabel("Амплитуда I₁", fontsize=12)
    plt.title(
        f"Зависимость амплитуды первой гармоники от Φ\n"
        f"I_avg={I_avg}, T0={T0}, T={T}, R0={R0}, R={R}, φ₀={phi0:.2f}, W={W}",
        fontsize=14,
    )
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в папку figures
    plt.savefig(f"figures/first_harmonic_intensity_I{I_avg}_T0{T0}_T{T}_R0{R0}_R{R}_phi{phi0:.2f}_W{W}.png")
    plt.show()


def plot_second_harmonic_intensity(T0, T, R0, R, phi0, W, Phi_max=10, I_avg=1.0):
    """
    Строит график зависимости I₂ от Phi для второй гармоники
    
    Параметры:
    T0 (float): коэффициент пропускания первого плеча
    T (float): коэффициент пропускания второго плеча
    R0 (float): коэффициент отражения первого плеча
    R (float): коэффициент отражения второго плеча
    phi0 (float): начальная фаза
    W (float): параметр W
    Phi_max (float): максимальное значение Phi для графика
    I_avg (float): средняя интенсивность
    """
    # Создаем папку figures, если её нет
    if not os.path.exists("figures"):
        os.makedirs("figures")

    # Создаем массив значений Phi
    Phi = np.linspace(0, Phi_max, 1000)

    # Вычисляем A2 и B2
    A2 = (T0 * T + R0 * R) * jv(2, W * Phi) / 2
    B2 = T0 * R0 * jv(2, Phi) / 2

    # Вычисляем C2
    C2 = 2 * I_avg * np.sqrt(A2**2 + B2**2 - 2 * A2 * B2 * np.cos((2 * W - 2) * phi0))

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(Phi, C2, "b-", linewidth=2)
    plt.xlabel("Φ", fontsize=12)
    plt.ylabel("Амплитуда I₂", fontsize=12)
    plt.title(
        f"Зависимость амплитуды второй гармоники от Φ\n"
        f"I_avg={I_avg}, T0={T0}, T={T}, R0={R0}, R={R}, φ₀={phi0:.2f}, W={W}",
        fontsize=14,
    )
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в папку figures
    plt.savefig(f"figures/second_harmonic_intensity_I{I_avg}_T0{T0}_T{T}_R0{R0}_R{R}_phi{phi0:.2f}_W{W}.png")
    plt.show()


plot_intensity_vs_phase(T=0.7, R=0.3)
plot_bessel_intensity(T=0.7, R=0.3, phi0=np.pi / 4)
plot_bessel2_intensity(T=0.7, R=0.3, phi0=np.pi / 4)
plot_first_harmonic_intensity(T0=0.7, T=0.7, R0=0.3, R=0.3, phi0=np.pi / 4, W=1.0)
plot_second_harmonic_intensity(T0=0.7, T=0.7, R0=0.3, R=0.3, phi0=np.pi / 4, W=1.0)
