Конечно, вот переработанная версия исследовательского документа. Я объединил теоретическую базу и базовый конвейер из первоначального проекта с подробными практическими шагами, примерами кода и, что самое важное, продвинутыми техниками для достижения максимальной производительности, такими как многопоточность, асинхронная GPU-синхронизация и интеграция с Python через DLPack, взятыми из предоставленных источников. В результате получилось более полное и полезное руководство, охватывающее весь конвейер от захвата до вывода на экран с акцентом на максимальную производительность и параллелизм.

***

## Исследование: Реализация высокопроизводительного конвейера захвата и обработки экрана на GPU в Windows

В данном документе представлено исследование и практическое руководство по реализации наиболее производительного конвейера для захвата экрана в Windows, который работает исключительно на графическом процессоре (GPU). Основная цель — захватить изображение, передать его напрямую в тензор PyTorch (или его эквивалент в C++ LibTorch) для обработки и затем отобразить результат, полностью исключив передачу данных через центральный процессор (CPU) на всех этапах [[1]](https://stackoverflow.com/questions/76656767/creating-an-initialization-and-capturing-function-that-returns-an-image-for-a-sc)[[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window) .

### 1. Обзор API для захвата экрана в Windows

Для высокопроизводительного захвата экрана в Windows существует несколько ключевых технологий [[3]](https://github.com/robmikh/Win32CaptureSample) . Выбор правильного API является первым и решающим шагом в построении эффективного конвейера [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .

*   **GDI (Graphics Device Interface):** Это устаревший метод, использующий такие функции, как `BitBlt`. Он сильно зависит от CPU, что приводит к копированию данных из видеопамяти в системную память и обратно [[1]](https://stackoverflow.com/questions/76656767/creating-an-initialization-and-capturing-function-that-returns-an-image-for-a-sc) . Для современных приложений, использующих аппаратное ускорение, этот метод часто неэффективен и может приводить к захвату черных окон . Он не подходит для нашей цели.
*   **Desktop Duplication API (DDA):** Часть DXGI, DDA долгое время был стандартом для высокопроизводительного захвата рабочего стола [[3]](https://github.com/robmikh/Win32CaptureSample) . Он позволяет приложению получить копию кадра в виде текстуры DirectX 11 (`ID3D11Texture2D`), и эта операция происходит полностью на GPU [[5]](https://windowsasusual.blogspot.com/2020/12/screen-capture-sample-clarified-few.html) . Основным недостатком является требование, чтобы приложение для захвата работало на том же GPU, что и отображаемый монитор, что усложняет работу на системах с несколькими видеокартами .
*   **Windows Graphics Capture (WGC):** Представленный в Windows 10 (версия 1803), WGC является современным и рекомендуемым API для захвата экрана . Он также предоставляет захваченный кадр в виде текстуры DirectX 11, но работает на более высоком уровне абстракции и имеет значительные преимущества:
    *   **Захват отдельных окон:** WGC позволяет надежно захватывать содержимое конкретного окна, даже если оно перекрыто другими окнами [[3]](https://github.com/robmikh/Win32CaptureSample) .
    *   **Кросс-адаптерная поддержка:** WGC без проблем работает на системах с несколькими GPU, что критически важно для сложных систем [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .
    *   **Безопасность:** API требует явного согласия пользователя на захват каждого окна или экрана, что повышает безопасность [[3]](https://github.com/robmikh/Win32CaptureSample)[[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .
    *   **Производительность:** В некоторых сценариях WGC может быть даже производительнее DDA, так как способен избежать дополнительной операции копирования данных на GPU [[3]](https://github.com/robmikh/Win32CaptureSample) .

**Вывод:** Для создания универсального и высокопроизводительного решения **Windows Graphics Capture (WGC)** является предпочтительным выбором благодаря своей гибкости, безопасности и отличной производительности [[3]](https://github.com/robmikh/Win32CaptureSample)[[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .

### 2. Архитектура асинхронного многопоточного конвейера

Для достижения максимальной производительности и полного параллелизма необходимо реализовать многопоточную архитектуру, где захват, обработка и отображение выполняются одновременно, не блокируя друг друга [ANSWER 1, ANSWER 2].

1.  **Поток Захвата (Capture Thread):** Отвечает за инициализацию DirectX 11 и WGC. Он захватывает кадры в виде текстур `ID3D11Texture2D`, копирует их в общую текстуру из пула и сигнализирует потоку обработки о доступности нового кадра с помощью `ID3D11Fence` [[6]](https://github.com/karwan5880/winrt_demo)[[7]](https://stackoverflow.com/questions/79498679/problems-with-windows-graphics-capture) .
2.  **Поток Обработки (Processing Thread):** Инициализирует CUDA и LibTorch. Он асинхронно ожидает сигнала от потока захвата на GPU. После получения сигнала он выполняет на GPU преобразование формата пикселей (например, с помощью CUDA-ядра), передает данные в модель LibTorch для инференса и сигнализирует о завершении [4, ANSWER 1].
3.  **Поток Отображения (Render Thread):** Управляет окном, обрабатывает пользовательский ввод и отображает результаты обработки, копируя их из общей текстуры в задний буфер цепочки обмена (`IDXGISwapChain`) [ANSWER 2].

Ключ к такой архитектуре — использование объектов синхронизации GPU-GPU, которые позволяют DirectX и CUDA координировать работу без блокировки потоков CPU [[5]](https://windowsasusual.blogspot.com/2020/12/screen-capture-sample-clarified-few.html)[[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window) .

### 3. Продвинутые детали реализации

#### 3.1. Сопоставление устройств DirectX и CUDA по LUID

В системе с несколькими GPU крайне важно, чтобы контексты DirectX и CUDA были созданы на одном и том же физическом устройстве, чтобы избежать медленной пересылки данных между GPU [[8]](https://www.youtube.com/watch?v=gQIG77PfLgo) . Самый надежный способ — сопоставление по локально-уникальному идентификатору (LUID) [[9]](https://www.rastertek.com/dx11win10tut03.html) .

1.  **Получение LUID адаптера DirectX:** От `ID3D11Device` получите `IDXGIAdapter`, а из его дескриптора (`DXGI_ADAPTER_DESC`) — поле `AdapterLuid` [[8]](https://www.youtube.com/watch?v=gQIG77PfLgo) .
2.  **Поиск соответствующего устройства CUDA:** Переберите все CUDA-устройства (`cudaGetDeviceCount`). Для каждого устройства получите его свойства (`cudaDeviceProp`), которые содержат поле `luid` .
3.  **Сравнение LUID:** Сравните LUID от DXGI с LUID каждого CUDA-устройства с помощью `memcmp`. При совпадении вы нашли нужный индекс устройства CUDA [[10]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html) .

Более простой метод предоставляет CUDA Runtime API — функция `cudaD3D11GetDevice()`, которая принимает `IDXGIAdapter` и возвращает индекс соответствующего CUDA-устройства [[3]](https://github.com/robmikh/Win32CaptureSample)[[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100)[[11]](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__D3D11.html)[[12]](https://stackoverflow.com/questions/77237723/how-to-render-direct3d-scene-to-texture-process-it-with-cuda-and-render-result) .

```cpp
// Пример с использованием cudaD3D11GetDevice
IDXGIDevice* pDXGIDevice;
d3d_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&pDXGIDevice);
IDXGIAdapter* pAdapter;
pDXGIDevice->GetAdapter(&pAdapter);

int cuda_device_id;
cudaError_t result = cudaD3D11GetDevice(&cuda_device_id, pAdapter);
if (result == cudaSuccess) {
    cudaSetDevice(cuda_device_id);
}
// ... освобождение pAdapter и pDXGIDevice ...
```

#### 3.2. Асинхронная синхронизация с `ID3D11Fence` и `cudaExternalSemaphore`

Для настоящей параллельной работы необходимо использовать `ID3D11Fence` (требует Windows 10 v1703+ и `ID3D11Device5`) и его эквивалент в CUDA — `cudaExternalSemaphore_t` [[13]](https://stackoverflow.com/questions/67909709/direct-write-to-d3d-texture-from-kernel)[[14]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html) . Это позволяет GPU-очереди CUDA ждать завершения команд в очереди DirectX без участия CPU [[5]](https://windowsasusual.blogspot.com/2020/12/screen-capture-sample-clarified-few.html)[[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window) .

**Схема синхронизации:**

1.  **Инициализация:**
    *   Создайте `ID3D11Fence` с флагом `D3D11_FENCE_FLAG_SHARED` .
    *   Создайте для него разделяемый дескриптор (`CreateSharedHandle`) [[13]](https://stackoverflow.com/questions/67909709/direct-write-to-d3d-texture-from-kernel) .
    *   В потоке CUDA импортируйте этот дескриптор в `cudaExternalSemaphore_t` с помощью `cudaImportExternalSemaphore`.

2.  **Конвейер (от захвата к обработке):**
    *   **Поток Захвата (DirectX):**
        1.  Получает кадр от WGC и копирует его в общую текстуру: `d3d_context->CopyResource(pSharedTexture, pWgcTexture);` [[6]](https://github.com/karwan5880/winrt_demo) .
        2.  Увеличивает значение счетчика для фенса: `UINT64 fenceValue = ++m_fenceValue;`.
        3.  Ставит в очередь DirectX команду сигнализации: `d3d_context->Signal(pFence, fenceValue);`. Эта команда выполнится на GPU после завершения копирования [[1]](https://stackoverflow.com/questions/76656767/creating-an-initialization-and-capturing-function-that-returns-an-image-for-a-sc)[[7]](https://stackoverflow.com/questions/79498679/problems-with-windows-graphics-capture) .
    *   **Поток Обработки (CUDA):**
        1.  Ставит в очередь CUDA команду ожидания: `cudaWaitExternalSemaphoresAsync(&m_cudaSemaphore, &waitParams, 1, stream);`. Эта команда заставит CUDA-поток ждать на GPU, пока DirectX не просигнализирует фенс [[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window) .
        2.  Сразу после `Wait` ставит в очередь `cudaGraphicsMapResources`, запуск CUDA-ядра, обработку LibTorch и `cudaGraphicsUnmapResources` [[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window)[[15]](https://forums.developer.nvidia.com/t/write-to-dx11-texture/63577) .

Эта схема обеспечивает конвейер: пока CUDA обрабатывает кадр N, DirectX уже может захватывать и копировать кадр N+1 [ANSWER 1].

#### 3.3. Эффективное преобразование формата пикселей на GPU

Кадры от WGC обычно приходят в формате `BGRA` (`DXGI_FORMAT_B8G8R8A8_UNORM`). Нейронные сети часто требуют формат `RGB` (или `BGR`), преобразованный в `float` и в компоновке `NCHW` (Количество, Каналы, Высота, Ширина) [[16]](https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor) . Выполнение этого на CPU — узкое место. Идеальное решение — CUDA-ядро, которое выполняет все преобразования за один проход [[3]](https://github.com/robmikh/Win32CaptureSample) .

**CUDA-ядро для `BGRA` -> `float RGB NCHW`:**

```cuda
__global__ void BgraToRgbNchw(cudaSurfaceObject_t inputSurface, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Чтение пикселя BGRA из текстуры DirectX (представленной как Surface Object)
    uchar4 pixel;
    surf2Dread(&pixel, inputSurface, x * sizeof(uchar4), y);

    // Нормализация  -> [0.0, 1.0] и перестановка каналов
    const float r = (float)pixel.z / 255.0f; // BGR'A' -> R
    const float g = (float)pixel.y / 255.0f; // B'G'RA -> G
    const float b = (float)pixel.x / 255.0f; // 'B'GRA -> B

    // Запись в NCHW формате (планарная компоновка)
    // Для N=1, C=3
    out[0 * height * width + y * width + x] = r; // Канал R
    out[1 * height * width + y * width + x] = g; // Канал G
    out[2 * height * width + y * width + x] = b; // Канал B
}
```
*   **Вход:** Использование `cudaSurfaceObject_t` для чтения из `cudaArray_t` (которое представляет текстуру DirectX) обеспечивает оптимальный доступ через кэш текстур [[17]](https://stackoverflow.com/questions/70268991/whats-pitch-in-cudamemcpy2dtoarray-and-cudamemcpy2dfromarray) .
*   **Выход:** Линейный буфер `out`, выделенный через `cudaMalloc`, затем может быть обернут в тензор LibTorch без копирования с помощью `torch::from_blob` [[16]](https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor)[[3]](https://github.com/robmikh/Win32CaptureSample)[[18]](https://developer.download.nvidia.com/compute/DevZone/docs/html/C/doc/html/group__CUDART__MEMORY_g1cc6e4eb2a5e0cd2bebbc8ebb4b6c46f.html)[[19]](https://forum.derivative.ca/t/libtorch-in-c/112501) .

#### 3.4. Обработка динамических изменений (изменение размера окна)

Когда пользователь изменяет размер или закрывает захватываемое окно, ресурсы, зависимые от размера (текстуры, буферы), становятся недействительными и должны быть пересозданы [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .

**Процедура обработки изменения размера:**

1.  **Обнаружение:** Отслеживайте событие `Closed` от `GraphicsCaptureItem` или `WM_SIZE` в цикле сообщений окна [[20]](https://discuss.pytorch.org/t/opengl-libtorch-and-cuda-interop-doing-inference-on-texture-data/165926)[[21]](https://discuss.pytorch.org/t/libtorch-memory-options-for-tensors-pinned-memory-zero-copy-memory/157420) . `Direct3D11CaptureFramePool` также требует вызова `Recreate()` при изменении размера [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .
2.  **Пауза и синхронизация:** Приостановите все потоки и дождитесь завершения текущих операций на GPU с помощью фенсов [ANSWER 1].
3.  **Освобождение ресурсов:**
    *   Отмените регистрацию ресурсов в CUDA: `cudaGraphicsUnregisterResource` [[22]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture?view=winrt-26100) .
    *   Освободите все COM-объекты, зависящие от размера: общие текстуры `ID3D11Texture2D`, `IDXGISwapChain` и т.д. [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .
4.  **Пересоздание ресурсов:**
    *   Вызовите `IDXGISwapChain::ResizeBuffers()` [[23]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html) .
    *   Вызовите `Direct3D11CaptureFramePool::Recreate()` с новым размером [[4]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture.graphicscapturepicker?view=winrt-26100) .
    *   Создайте заново пул общих текстур и буферы CUDA с новыми размерами.
5.  **Повторная инициализация:**
    *   Заново зарегистрируйте новые текстуры в CUDA: `cudaGraphicsD3D11RegisterResource` [[3]](https://github.com/robmikh/Win32CaptureSample)[[22]](https://learn.microsoft.com/en-us/uwp/api/windows.graphics.capture?view=winrt-26100) .
    *   Возобновите работу конвейера.

### 4. Интеграция с Python: Zero-Copy передача в PyTorch

Прямой способ получить нативный указатель на текстуру DirectX в Python и создать из него тензор PyTorch без копирования через CPU с помощью стандартных библиотек **не существует** [[15]](https://forums.developer.nvidia.com/t/write-to-dx11-texture/63577)[[10]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html)[[24]](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-c-programming-guide/index.html) . `ID3D11Texture2D` — это непрозрачный объект, часто с нелинейной (tiled/swizzled) компоновкой памяти, с которой `torch.from_blob` напрямую работать не может [[25]](https://stackoverflow.com/questions/19871443/how-do-you-capture-current-frame-from-a-mediaelement-in-winrt-8-1)[[26]](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html) .

Наиболее эффективное решение — создать **собственное C++ расширение для Python** (с помощью `pybind11`), которое использует стандарт **DLPack** для обмена тензорами между фреймворками без копирования данных [[27]](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-copyresource)[[28]](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copyresource)[[23]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html) .

**Механизм: C++ + Pybind11 + DLPack**

1.  **C++ Backend:** Напишите C++ модуль, который реализует весь низкоуровневый конвейер:
    *   Инициализация WGC, DirectX, CUDA [[10]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html) .
    *   Захват кадра, копирование в общую текстуру, синхронизация через `ID3D11Fence`.
    *   Выполнение CUDA-ядра, которое копирует и преобразует данные из `cudaArray_t` в **линейный** буфер на GPU (`cudaMalloc`) [[16]](https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor) .

2.  **Экспорт в Python через DLPack:**
    *   Создайте C++ функцию, которая будет вызываться из Python.
    *   Эта функция выполняет шаги выше и получает указатель на линейный буфер GPU (`void* cudaPtr`).
    *   Она создает структуру `DLManagedTensor`, заполняя ее информацией о тензоре: указатель на данные, устройство (`kDLCUDA`), тип данных, форма, шаги (strides) [[29]](https://forums.developer.nvidia.com/t/cudagraphicsmapresources-each-frame-or-just-once-when-cuda-opengl-interop-which-better/275015) .
    *   Оборачивает `DLManagedTensor` в `PyCapsule`. **Ключевой момент:** в деструктор (`deleter`) капсулы помещается лямбда-функция, которая освобождает все связанные ресурсы (кадр WGC, CUDA-буфер и т.д.), когда тензор PyTorch будет удален сборщиком мусора Python. Это обеспечивает корректное управление жизненным циклом [[30]](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-mapping)[[29]](https://forums.developer.nvidia.com/t/cudagraphicsmapresources-each-frame-or-just-once-when-cuda-opengl-interop-which-better/275015) .
    *   Функция возвращает `PyCapsule` в Python.

3.  **Использование в Python:**
    *   Импортируйте ваше C++ расширение.
    *   Вызовите функцию, которая возвращает `PyCapsule`.
    *   Создайте тензор PyTorch с помощью `torch.utils.dlpack.from_dlpack(capsule)`. Копирования данных через CPU на этом этапе не происходит [[23]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html)[[24]](https://docs.nvidia.com/cuda/archive/11.4.1/cuda-c-programming-guide/index.html) .

```python
# Псевдокод использования в Python
import torch
import your_custom_wgc_module as wgc # Ваша C++ библиотека, собранная с pybind11

# Инициализация захвата через ваш модуль
wgc.initialize_capture(target_hwnd)

# В цикле получаем капсулу, которая содержит указатель на GPU-память
# Копирования через CPU здесь нет
gpu_frame_capsule = wgc.capture_frame_as_dlpack()

if gpu_frame_capsule:
    # PyTorch создает тензор, который "смотрит" в ту же память на GPU
    pytorch_tensor_gpu = torch.utils.dlpack.from_dlpack(gpu_frame_capsule)

    # Теперь pytorch_tensor_gpu готов для обработки на GPU
    print(pytorch_tensor_gpu.shape, pytorch_tensor_gpu.device)

# Когда `pytorch_tensor_gpu` выходит из области видимости и удаляется,
# автоматически вызывается деструктор в C++, который освобождает ресурсы.
```

### Резюме (Executive Summary)

Для реализации наиболее производительного конвейера захвата, обработки и отображения экрана исключительно на GPU в Windows рекомендуется следующий асинхронный многопоточный подход:

*   **Архитектура:** Разделить приложение на **три параллельных потока** (захват, обработка, отображение), работающих независимо и обменивающихся данными через пул общих текстур DirectX 11 [ANSWER 1, ANSWER 2].

*   **Захват:** Использовать современный **Windows Graphics Capture (WGC) API** для получения кадра в виде временной текстуры `ID3D11Texture2D` [[3]](https://github.com/robmikh/Win32CaptureSample) .

*   **Синхронизация:** Для координации между потоками DirectX и CUDA использовать механизм синхронизации GPU-GPU: **`ID3D11Fence`** и **`cudaExternalSemaphore_t`**. Это позволяет избежать блокировок CPU и достичь максимального параллелизма [[5]](https://windowsasusual.blogspot.com/2020/12/screen-capture-sample-clarified-few.html)[[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window)[[7]](https://stackoverflow.com/questions/79498679/problems-with-windows-graphics-capture)[[13]](https://stackoverflow.com/questions/67909709/direct-write-to-d3d-texture-from-kernel) .

*   **Передача в обработку (C++ LibTorch):**
    1.  В потоке захвата копировать кадр WGC в общую текстуру (`CopyResource`) и сигнализировать `ID3D11Fence` [[6]](https://github.com/karwan5880/winrt_demo)[[7]](https://stackoverflow.com/questions/79498679/problems-with-windows-graphics-capture) .
    2.  В потоке обработки асинхронно ждать сигнала на `cudaExternalSemaphore_t`, затем отображать (`map`) ресурс в CUDA [[2]](https://stackoverflow.com/questions/11283015/screen-capture-specific-window) .
    3.  Выполнить преобразование формата пикселей (например, `BGRA` в `NCHW float`) с помощью **CUDA-ядра** для максимальной производительности [[16]](https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor)[[3]](https://github.com/robmikh/Win32CaptureSample) .
    4.  Обернуть полученный линейный буфер в тензор `torch::Tensor` с помощью `torch::from_blob` [[14]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html)[[15]](https://forums.developer.nvidia.com/t/write-to-dx11-texture/63577) .

*   **Передача в обработку (Python PyTorch):**
    1.  Реализовать низкоуровневую логику в **C++ расширении** с использованием `pybind11` [[27]](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-copyresource)[[10]](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html) .
    2.  Передавать обработанный на GPU линейный буфер в Python через стандарт **DLPack**, обернув его в `PyCapsule` с кастомным деструктором для управления ресурсами [[30]](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-mapping)[[28]](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copyresource)[[29]](https://forums.developer.nvidia.com/t/cudagraphicsmapresources-each-frame-or-just-once-when-cuda-opengl-interop-which-better/275015) .
    3.  В Python создавать тензор с помощью `torch.utils.dlpack.from_dlpack`, что гарантирует **отсутствие копирования данных через CPU** [[23]](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/graphics-interop.html) .

*   **Отображение результата:** В потоке отображения копировать обработанную текстуру из общего пула в задний буфер цепочки обмена (`IDXGISwapChain`) с помощью `CopyResource` и выводить на экран (`Present`) [[7]](https://stackoverflow.com/questions/79498679/problems-with-windows-graphics-capture) .

Этот конвейер полностью исключает передачу данных между GPU и CPU, минимизирует задержки за счет асинхронной архитектуры и является наиболее производительным решением для поставленной задачи.