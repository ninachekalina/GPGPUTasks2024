#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// Функция для вычисления множества Мандельброта
__kernel void mandelbrot(
    __global float* results,   // Массив для хранения результатов
    unsigned int width,        // Ширина изображения
    unsigned int height,       // Высота изображения
    float fromX,               // Начальная координата X
    float fromY,               // Начальная координата Y
    float sizeX,               // Размер по оси X
    float sizeY,               // Размер по оси Y
    unsigned int iters,        // Количество итераций
    int smoothing              // Включение/выключение сглаживания
)
{
    const float threshold = 256.0f;       // Порог для проверки выхода за границы
    const float threshold2 = threshold * threshold; // Квадрат порога

    // Получаем глобальные индексы для работы с пикселями
    int i = get_global_id(0);
    int j = get_global_id(1);

    // Преобразуем координаты пикселя в значения комплексной плоскости
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    // Инициализация значений для комплексного числа
    float x = x0;
    float y = y0;

    int iter = 0;
    // Основной цикл вычислений для множества Мандельброта
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;  // Итерации для X
        y = 2.0f * xPrev * y + y0;  // Итерации для Y
        // Проверка на выход за порог
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    // Результат итераций
    float result = iter;

    // Если сглаживание включено, применяем его
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    // Нормализация результата
    result = result / iters;
    results[j * width + i] = result; // Сохраняем результат
}
