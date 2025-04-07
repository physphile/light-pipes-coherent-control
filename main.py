import LightPipes as lp
import numpy as np
import matplotlib.pyplot as plt


def interference_on_gold(wavelength):
    """
    Функция для моделирования интерференции двух пучков на золотой плёнке.
    Аргумент: wavelength — длина волны в нанометрах (например, 500 для 500 нм).
    Возвращает: двумерный массив интенсивности.
    """
    # Перевод длины волны в мм для LightPipes
    wavelength_mm = wavelength * 1e-6  # нм → мм

    # Параметры системы
    size = 10  # размер сетки в мм
    N = 500  # количество точек
    z = 50  # расстояние до плёнки в мм

    # Инициализация поля
    F = lp.Begin(size, wavelength_mm, N)

    # Первый пучок (референтный)
    F1 = lp.GaussAperture(1.0, 0, 0, 1, F)  # Гауссов пучок
    F1 = lp.Fresnel(z, F1)  # Распространение до плёнки

    # Второй пучок (с модулятором)
    F2 = lp.GaussAperture(1.0, 0, 0, 1, F)
    x = np.linspace(-size / 2, size / 2, N)
    X, Y = np.meshgrid(x, x)
    phase = 1.0 * np.sin(2 * np.pi * 0.1 * X)  # Синусоидальная фаза (амплитуда 1 рад)
    F2.field *= np.exp(1j * phase)  # Применяем фазовый сдвиг напрямую к полю
    F2 = lp.Fresnel(z, F2)  # Распространение до плёнки

    # Учёт отражения от золотой плёнки
    # Примерная зависимость n и k от длины волны
    wavelengths = np.array([400, 500, 600, 700])  # нм
    n_r = np.array([1.64, 0.47, 0.25, 0.17])  # действительная часть
    k = np.array([1.96, 2.39, 3.15, 3.80])  # мнимая часть
    n_gold_r = np.interp(wavelength, wavelengths, n_r)
    n_gold_k = np.interp(wavelength, wavelengths, k)
    n_gold = n_gold_r + 1j * n_gold_k

    # Коэффициент отражения для нормального падения
    r = (1 - n_gold) / (1 + n_gold)
    F1.field *= r  # Применяем отражение к первому пучку
    F2.field *= r  # Применяем отражение ко второму пучку

    # Интерференция отражённых пучков
    F_combined = lp.BeamMix(F1, F2)

    # Вычисление интенсивности
    I = lp.Intensity(1, F_combined)
    return I


def plot_interference(intensity, wavelength):
    """
    Функция для построения графика интерференционной картины.
    Аргументы:
        intensity — двумерный массив интенсивности.
        wavelength — длина волны в нанометрах (для подписи графика).
    """
    plt.imshow(intensity, cmap="hot")
    plt.title(f"Интерференция на золотой плёнке при λ = {wavelength} нм")
    plt.colorbar(label="Интенсивность")
    plt.show()


def plot_intensity_vs_phase(wavelength, phase_range=np.linspace(0, 2*np.pi, 100)):
    """
    Функция для построения графика зависимости интенсивности от фазового сдвига.
    Аргументы:
        wavelength — длина волны в нанометрах
        phase_range — массив значений фазового сдвига
    """
    intensities = []
    for phase_shift in phase_range:
        # Инициализация поля
        size = 10
        N = 500
        z = 50
        wavelength_mm = wavelength * 1e-6
        
        F = lp.Begin(size, wavelength_mm, N)
        
        # Первый пучок (референтный)
        F1 = lp.GaussAperture(1.0, 0, 0, 1, F)
        F1 = lp.Fresnel(z, F1)
        
        # Второй пучок с переменным фазовым сдвигом
        F2 = lp.GaussAperture(1.0, 0, 0, 1, F)
        F2.field *= np.exp(1j * phase_shift)  # Применяем фазовый сдвиг
        F2 = lp.Fresnel(z, F2)
        
        # Учёт отражения от золотой плёнки
        wavelengths = np.array([400, 500, 600, 700])
        n_r = np.array([1.64, 0.47, 0.25, 0.17])
        k = np.array([1.96, 2.39, 3.15, 3.80])
        n_gold_r = np.interp(wavelength, wavelengths, n_r)
        n_gold_k = np.interp(wavelength, wavelengths, k)
        n_gold = n_gold_r + 1j * n_gold_k
        
        r = (1 - n_gold) / (1 + n_gold)
        F1.field *= r
        F2.field *= r
        
        # Интерференция и измерение интенсивности в центре
        F_combined = lp.BeamMix(F1, F2)
        I = lp.Intensity(1, F_combined)
        intensities.append(I[N//2, N//2])  # Интенсивность в центре
    
    plt.figure(figsize=(10, 6))
    plt.plot(phase_range, intensities)
    plt.title(f"Зависимость интенсивности от фазового сдвига при λ = {wavelength} нм")
    plt.xlabel("Фазовый сдвиг (рад)")
    plt.ylabel("Интенсивность")
    plt.grid(True)
    plt.show()


plot_intensity_vs_phase(628)  # Для длины волны 628 нм
