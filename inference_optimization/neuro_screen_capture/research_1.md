Архитектура и реализация пайплайна сверхнизкой задержки для захвата и нейросетевой обработки изображений на GPU в среде Windows
Аннотация
В данном фундаментальном исследовательском отчете представлен исчерпывающий технический анализ, архитектурное проектирование и детали реализации высокопроизводительного конвейера обработки видеоданных в реальном времени на платформе Windows. Ключевой задачей исследования является полное исключение центрального процессора (CPU) и системной оперативной памяти (RAM) из тракта передачи данных между подсистемой захвата экрана и подсистемой нейросетевого инференса. Реализуемая парадигма Zero-Copy, опирающаяся на прямое взаимодействие (Interop) между DirectX 11 и CUDA, позволяет достичь теоретического минимума задержки, ограниченного лишь физическими параметрами дисплея и временем выполнения вычислительных ядер GPU.
В отчете детально рассматриваются механизмы работы Windows Display Driver Model (WDDM), сравниваются API захвата DXGI Desktop Duplication и Windows Graphics Capture, анализируются низкоуровневые аспекты управления памятью видеокарты (тайловая против линейной организации), а также предлагаются оптимизированные алгоритмы синхронизации потоков с использованием Keyed Mutex. Особое внимание уделено интеграции с библиотекой LibTorch (C++ API PyTorch) для обеспечения бесшовного перехода от графических ресурсов к тензорным вычислениям. Результатом исследования является спецификация базового пайплайна: Захват -> Конвертация CUDA -> Тензор PyTorch -> Отрисовка, работающего в замкнутом цикле на графическом ускорителе.
Глава 1. Теоретические основы и проблематика латентности в системах реального времени
1.1. Постановка задачи и критический путь данных
В задачах компьютерного зрения, требующих мгновенной реакции — будь то системы автопилотирования, игровые ассистенты, работающие на частоте обновления монитора (144 Гц/240 Гц и выше), или системы дополненной реальности — критическим параметром качества является латентность (latency) или задержка "glass-to-glass" (от фотона на экране до фотона реакции).
Традиционная архитектура обработки изображений в Windows строится по циклической схеме, вовлекающей множество пересылок данных:
VRAM -> RAM (Readback): Драйвер видеокарты копирует буфер кадра через шину PCIe в системную память.
CPU Processing: Процессор преобразует формат пикселей (часто с использованием медленных циклов или SIMD-инструкций).
RAM -> VRAM (Upload): Подготовленные данные (тензоры) загружаются обратно в память GPU для инференса нейросети.
Анализ пропускной способности показывает фундаментальную неэффективность такого подхода.1 Пропускная способность шины PCIe 4.0 x16 составляет теоретически 32 ГБ/с (фактически около 26 ГБ/с). Передача кадра 4K (3840x2160, RGBA, 32 бита) занимает около 33 МБ. Теоретическое время передачи составляет ~1.2 мс, однако накладные расходы на инициализацию транзакций DMA, переключение контекста ядра ОС и синхронизацию кэшей CPU увеличивают это время до 5-10 мс. В двунаправленном цикле (Readback + Upload) потери могут составлять до 15-20 мс, что уже превышает бюджет времени кадра для монитора 60 Гц (16.6 мс).
Для достижения максимальной производительности необходимо реализовать архитектуру GPU-Residency, где данные пикселей, будучи рожденными в VRAM (композитором Windows DWM), никогда не покидают кристалл видеокарты вплоть до момента их отображения после обработки.
1.2. Архитектура WDDM и роль DWM
Понимание того, как Windows формирует изображение, критически важно для выбора точки захвата. Desktop Window Manager (DWM) — это композитор окон, работающий поверх DirectX. Каждое приложение рисует свое содержимое в свои off-screen поверхности. DWM собирает эти поверхности и компонует итоговый образ рабочего стола (Desktop Image) в единый буфер в видеопамяти.
Именно этот финальный буфер является целью захвата. Существует два основных способа получить к нему доступ:
GDI (BitBlt): Устаревший механизм, использующий CPU и системную память. Абсолютно непригоден для Zero-Copy.
DirectX API (DXGI / WGC): Механизмы, позволяющие получить дескриптор (handle) ресурса непосредственно в видеопамяти.
Модель WDDM (Windows Display Driver Model) версий 2.x и выше вводит понятие "общей памяти" (Shared Memory) и межпроцессного взаимодействия ресурсов GPU. Это позволяет одному процессу (например, нашему приложению) открывать ресурсы, созданные другим процессом (DWM), без копирования самих данных, оперируя лишь ссылками и таблицами страниц памяти GPU.
Глава 2. Сравнительный анализ API захвата: DDA против WGC
Выбор правильного API захвата определяет фундамент производительности всего приложения. На данный момент в экосистеме Windows доминируют два подхода: DXGI Desktop Duplication API (DDA) и Windows Graphics Capture (WGC).
2.1. DXGI Desktop Duplication API (DDA)
Введенный в Windows 8, DDA является частью инфраструктуры DXGI. Он предоставляет доступ к обновлениям рабочего стола на уровне драйвера.
Механизм работы: Приложение запрашивает у выхода (IDXGIOutput) дубликацию через DuplicateOutput. Возвращаемый интерфейс IDXGIOutputDuplication предоставляет метод AcquireNextFrame. Этот метод блокирует выполнение до появления нового кадра или истечения таймаута.1 Важнейшей особенностью DDA является то, что возвращаемый ресурс IDXGIResource фактически указывает на внутренний буфер, который DWM уже подготовил для вывода на монитор.

Характеристика
Описание в контексте DDA
Доступ к памяти
Прямой доступ к VRAM через ID3D11Texture2D.
Формат пикселей
Строго DXGI_FORMAT_B8G8R8A8_UNORM.3
Метаданные
Предоставляет DirtyRects (измененные области) и MoveRects (перемещенные области).4
Латентность
Минимально возможная, так как нет промежуточных слоев абстракции WinRT.
Совместимость
Требует, чтобы монитор был подключен к тому же GPU, на котором запущен код захвата.5

Анализ производительности DDA: Исследования показывают, что DDA обеспечивает наиболее стабильный фреймрейт при захвате всего экрана. Поскольку он работает синхронно с обновлением монитора (V-Sync драйвера), он гарантирует получение каждого кадра. Отсутствие лишних копий внутри подсистемы захвата делает его идеальным кандидатом для Zero-Copy пайплайнов.7
2.2. Windows Graphics Capture (WGC)
WGC — это современный объектно-ориентированный API (пространство имен Windows.Graphics.Capture), представленный в Windows 10 версии 1803.
Механизм работы: WGC использует концепцию Direct3D11CaptureFramePool. Приложение создает пул буферов, и система (через службу CaptureService) асинхронно заполняет эти буферы при обновлении целевого окна или экрана. Событие FrameArrived сигнализирует о готовности данных.8
Преимущества и недостатки:
Плюсы: WGC отлично справляется с захватом отдельных окон, даже если они перекрыты другими окнами. Он не требует прав администратора для захвата некоторых защищенных окон (в отличие от старых методов).
Минусы: В сценариях захвата полного экрана WGC часто показывает худшую производительность по сравнению с DDA, особенно если в системе не включен режим HAGS (Hardware Accelerated GPU Scheduling). Некоторые бенчмарки указывают на нестабильность времени кадра (jitter) и дополнительные накладные расходы на копирование внутри Direct3D11CaptureFramePool.6 Также WGC требует наличия цикла обработки сообщений (DispatcherQueue) или использования FreeThreaded модели, что усложняет интеграцию в чистые C++ консольные приложения.
2.3. Итоговый выбор
Для задачи "максимально быстрого захвата экрана" (Full Screen) с последующей обработкой, DXGI Desktop Duplication API (DDA) является безальтернативным выбором.2 Его недостатки (проблемы с курсором, привязка к GPU) в данном контексте несущественны по сравнению с преимуществом в "сырой" скорости доступа к VRAM. WGC вводит дополнительные слои абстракции WinRT, которые не нужны для задачи захвата всего десктопа и могут вносить микрозадержки.6
Глава 3. Архитектура Zero-Copy: Взаимодействие DirectX 11 и CUDA
После получения текстуры от DDA, критической задачей становится передача данных в контекст CUDA без копирования через системную шину. Это достигается через механизм CUDA-Direct3D 11 Interoperability.
3.1. Регистрация ресурсов и стратегия двойной буферизации
Функция cudaGraphicsD3D11RegisterResource позволяет зарегистрировать ресурс D3D11 для использования в CUDA. Однако здесь кроется подводный камень производительности: вызов регистрации является чрезвычайно тяжелой операцией, вызывающей синхронизацию драйвера и полную остановку конвейера.11
Проблема: DDA при каждом вызове AcquireNextFrame может возвращать ссылку на разные ресурсы из внутреннего пула swap-chain DWM. Если мы будем регистрировать полученную текстуру каждый кадр, производительность упадет до неприемлемых значений (десятки миллисекунд).
Решение (Staging Texture Strategy):
Необходимо реализовать стратегию с постоянным промежуточным буфером:
При инициализации приложения создается одна постоянная текстура (shared_texture) с флагами D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET.
Эта текстура регистрируется в CUDA один раз (cudaGraphicsD3D11RegisterResource).
В горячем цикле (Hot Loop):
Получаем кадр от DDA (desktop_resource).
Копируем содержимое desktop_resource в shared_texture используя ID3D11DeviceContext::CopyResource.
Освобождаем кадр DDA.
Операция CopyResource выполняется внутри GPU (Device-to-Device Copy) и использует широчайшую внутреннюю шину памяти видеокарты (сотни ГБ/с). Задержка этой операции пренебрежимо мала (< 0.1 мс для 4K на современных GPU) по сравнению с накладными расходами на перерегистрацию ресурса.13
3.2. Маппинг ресурсов и проблемы синхронизации
Для доступа к данным в CUDA используется пара функций cudaGraphicsMapResources и cudaGraphicsUnmapResources.
Map: Гарантирует, что любые предыдущие команды DirectX, использующие этот ресурс, завершены. Это неявная точка синхронизации.15
Unmap: Сигнализирует DirectX, что CUDA завершила работу с ресурсом.
Внутри маппированной области ресурс доступен как cudaArray_t (через cudaGraphicsSubResourceGetMappedArray). Важно понимать, что cudaArray представляет собой текстуру с тайловой (tiled) организацией памяти (часто Z-order curve). Это оптимизировано для 2D-локальности текстурных выборок, но несовместимо с линейным (Row-Major) доступом, который ожидают стандартные C++ указатели или тензоры PyTorch.16
Глава 4. Ядро CUDA: Конвертация и Препроцессинг
Это "сердце" системы производительности. Нам необходимо трансформировать данные из cudaArray (BGRA, uint8, Tiled) в линейный буфер (RGB, float32, Linear), который сможет прочитать PyTorch. Стандартный cudaMemcpy2DFromArray может выполнить линеаризацию, но он не умеет менять порядок каналов и нормализовать данные, что потребует второго прохода. Мы объединим все в одно кастомное ядро.
4.1. Архитектура ядра конвертации
Мы будем использовать Texture Object API в CUDA для чтения данных. Текстурный блок (Texture Unit) GPU аппаратно ускоряет чтение из тайловой памяти, обеспечивая высокую эффективность кэша.16
Спецификация ядра:
Вход: cudaTextureObject_t (привязанный к cudaArray из D3D11 текстуры).
Выход: float* (указатель на глобальную память, выделенную через cudaMalloc).
Операции: Чтение, Swizzling (BGRA -> RGB), Normalization (/255.0), запись в NCHW или NHWC раскладку.
Пример реализации ядра (Pseudo-C++):

C++


__global__ void preprocess_kernel(cudaTextureObject_t texObj, float* out_tensor, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Аппаратное чтение с использованием текстурного кэша
        // tex2D<uchar4> возвращает вектор из 4 байт (x=B, y=G, z=R, w=A для формата DXGI_FORMAT_B8G8R8A8_UNORM)
        uchar4 pixel = tex2D<uchar4>(texObj, x, y);

        // Индекс для планарного формата (NCHW), который предпочитает PyTorch
        // RRR... GGG... BBB...
        int area = width * height;
        int idx_r = y * width + x;
        int idx_g = area + idx_r;
        int idx_b = 2 * area + idx_r;

        // Конвертация, перестановка и нормализация
        // B (x) -> Blue channel
        // G (y) -> Green channel
        // R (z) -> Red channel
        
        // Важно: Порядок каналов в тензоре обычно RGB. 
        // DXGI дает BGRA. Значит pixel.z (R) идет в первый план, pixel.y (G) во второй, pixel.x (B) в третий.
        
        out_tensor[idx_r] = pixel.z / 255.0f;
        out_tensor[idx_g] = pixel.y / 255.0f;
        out_tensor[idx_b] = pixel.x / 255.0f;
    }
}


Такой подход (Kernel Fusion) экономит пропускную способность памяти, так как мы читаем данные из VRAM только один раз и пишем один раз, выполняя все трансформации в регистрах SM (Streaming Multiprocessor).19
4.2. Оптимизация параметров запуска
Для максимальной загрузки (occupancy) GPU следует выбирать размер блока потоков (block size) кратным размеру варпа (32). Типичные значения: 16x16 (256 потоков) или 32x8. Сетка (Grid) рассчитывается как (width + block.x - 1) / block.x, (height + block.y - 1) / block.y.
Глава 5. Интеграция с LibTorch (PyTorch C++)
Использование C++ API PyTorch (LibTorch) является обязательным требованием для исключения накладных расходов Python (GIL, интерпретация) в критическом цикле.
5.1. Создание тензора без копирования (torch::from_blob)
Функция torch::from_blob является ключевым инструментом для Zero-Copy интеграции. Она создает объект torch::Tensor, который ссылается на уже существующую память, не владея ею.20
Управление памятью:
Аллокация: Память под тензор (float* gpu_buffer) выделяется через cudaMalloc один раз при инициализации.
Обертка: В каждом кадре, после работы ядра конвертации, мы создаем тензор:
C++
auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
torch::Tensor input = torch::from_blob(gpu_buffer, {1, 3, height, width}, options);

Это операция константного времени, создающая лишь структуру метаданных тензора.
Безопасность: Важно гарантировать, что gpu_buffer не будет освобожден, пока input используется. Поскольку мы управляем циклом вручную, это легко обеспечить.
5.2. Проблемы со страйдами (Strides) и выравниванием
При выделении памяти через cudaMallocPitch (для выравнивания строк по 256/512 байт для старых GPU) возникают "дырки" в памяти в конце каждой строки. PyTorch поддерживает страйды (strides), и их можно передать в from_blob. Однако, нейросети обычно требуют плотной упаковки (contiguous) каналов. Если использовать cudaMallocPitch, тензор будет иметь разрывы, и многие операторы PyTorch (например, conv2d в cuDNN) могут отказать в работе или молча выполнить копирование в плотный буфер (.contiguous()), что убьет производительность.21
Рекомендация: Использовать обычный cudaMalloc (линейная память без паддинга строк) для буфера тензора. Наше кастомное ядро конвертации само разложит пиксели плотно (строка за строкой без пропусков). Это гарантирует совместимость с любыми операторами PyTorch без дополнительных копий.
Глава 6. Синхронизация: Fences vs Keyed Mutex
Для корректной работы конвейера "DirectX -> CUDA -> DirectX" необходима жесткая, но быстрая синхронизация. Использование cudaDeviceSynchronize() недопустимо, так как это заставляет CPU ждать GPU.
6.1. Проблема: Блокировка GPU
Если просто вызвать CopyResource (DirectX) и затем сразу запустить CUDA Kernel, драйвер может не гарантировать порядок выполнения, если они находятся в разных очередях команд, или, наоборот, будет чрезмерно сериализовывать их.
6.2. Решение: IDXGIKeyedMutex
Для общих ресурсов (D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) механизм Keyed Mutex предоставляет аппаратную синхронизацию.23
Алгоритм:
У ресурса есть "ключ" (key).
DirectX: Вызывает AcquireSync(0). Это команда GPU: "Жди, пока ключ не станет 0". После захвата выполняет CopyResource. Затем ReleaseSync(1): "Установи ключ в 1".
CUDA: Использует External Semaphore API для импорта мьютекса.
cudaWaitExternalSemaphoresAsync: Ждет ключ 1.
Выполняет ядра.
cudaSignalExternalSemaphoresAsync: Устанавливает ключ 0.
Этот механизм работает полностью на стороне GPU. CPU лишь ставит команды "ждать" и "сигналить" в буфер команд. Это обеспечивает истинный асинхронный пайплайн.25
Глава 7. Отрисовка результатов (Rendering Back)
После того, как тензор прошел через нейросеть (или заглушку), его нужно отобразить на экране.
7.1. Обратное ядро конвертации
Создается второе CUDA ядро: Linear Tensor (RGB Float) -> Texture (BGRA UInt8).
Вход: float* тензора.
Выход: cudaSurfaceObject_t. Используется Surface API (surf2Dwrite), так как мы пишем в текстуру. Texture Object API — read-only.13
В ядре происходит денормализация (val * 255) и clamp (обрезание значений, выходящих за ).
7.2. Презентация (Presentation)
Результат записывается в ту же "Shared Texture" (или вторую, для двойной буферизации).
DirectX копирует из Shared Texture в BackBuffer SwapChain'а окна визуализации.
Вызывается IDXGISwapChain::Present.
Режимы SwapChain:
Для минимизации задержки следует использовать модель DXGI_SWAP_EFFECT_FLIP_DISCARD. В методе Present(SyncInterval, Flags):
SyncInterval = 0: Отключает V-Sync. Мгновенный вывод, возможен тиринг.
Flags = DXGI_PRESENT_ALLOW_TEARING: Позволяет выводить кадр, даже если монитор еще не обновился (требует поддержки в создании SwapChain и настроек монитора FreeSync/G-Sync).
Глава 8. Итоговая спецификация пайплайна (Детальный алгоритм)
В данном разделе сводятся воедино все компоненты в единый алгоритм работы приложения.
8.1. Инициализация (Startup)
D3D11 Setup: Создать ID3D11Device, ID3D11DeviceContext.
Capture Setup: Получить IDXGIOutputDuplication через DuplicateOutput.
Shared Resource: Создать текстуру shared_tex (RGBA, Bind: RenderTarget+ShaderResource, Misc: SharedKeyedMutex).
CUDA Setup: Инициализировать контекст CUDA. Получить дескриптор устройства из D3D адаптера (cudaD3D11GetDevice).
Interop:
Открыть shared_tex в CUDA через cudaImportExternalMemory (если используем новый API) или cudaGraphicsD3D11RegisterResource (Legacy, но проще).
Для Keyed Mutex: Импортировать семафоры через cudaImportExternalSemaphore.
Memory:
cudaMalloc для входного тензора (in_ptr).
cudaMalloc для выходного тензора (out_ptr).
Objects: Создать cudaTextureObject (для чтения из shared_tex) и cudaSurfaceObject (для записи в shared_tex или отдельную выходную текстуру).
8.2. Рабочий цикл (Run Loop)
Цикл выполняется бесконечно.
Захват (D3D):
Duplication->AcquireNextFrame(&frame_info, &desktop_res).
Если таймаут — пропустить итерацию.
KeyedMutex->AcquireSync(0). (Захват GPU прав на текстуру).
Context->CopyResource(shared_tex, desktop_res). (Копирование кадра).
KeyedMutex->ReleaseSync(1). (Передача прав CUDA).
Duplication->ReleaseFrame().
Обработка (CUDA):
cudaWaitExternalSemaphoresAsync(..., 1). (Ждем данные от D3D).
Kernel 1: texture_to_tensor<<<...>>>(tex_obj, in_ptr). (BGRA -> RGB Float).
PyTorch:
Создать in_tensor = torch::from_blob(in_ptr,...).
out_tensor = model.forward(in_tensor). (Или out_tensor = in_tensor для теста).
Важно: Если модель меняет страйды или память, убедиться, что out_tensor указывает на out_ptr или данные копируются в него.
Kernel 2: tensor_to_surface<<<...>>>(out_ptr, surf_obj). (RGB Float -> BGRA).
cudaSignalExternalSemaphoresAsync(..., 0). (Возвращаем права D3D).
Отрисовка (D3D):
KeyedMutex->AcquireSync(0). (Ждем завершения CUDA).
Context->CopyResource(backbuffer, shared_tex). (Копируем результат в окно).
KeyedMutex->ReleaseSync(1).
SwapChain->Present(0, DXGI_PRESENT_ALLOW_TEARING).
Глава 9. Заключение
Предложенная архитектура решает задачу сверхбыстрого захвата и обработки экрана, полностью устраняя узкие места классических подходов.
Ключевые достижения архитектуры:
Zero-Copy Host Memory: Данные пикселей ни разу не попадают в оперативную память (RAM).
Минимальная синхронизация: Использование Keyed Mutex позволяет CPU работать асинхронно, не блокируясь на ожидании завершения команд GPU.
Оптимизация форматов: Кастомные ядра CUDA выполняют конвертацию форматов "на лету", используя высокую пропускную способность L1/L2 кэшей GPU, вместо медленных операций глобальной памяти.
Совместимость с Deep Learning: Прямая интеграция с LibTorch через from_blob позволяет использовать всю мощь экосистемы PyTorch без накладных расходов на ввод-вывод.
Данный пайплайн является эталонным решением для построения систем компьютерного зрения реального времени на платформе Windows и обеспечивает задержку, близкую к теоретическому аппаратному минимуму.

Компонент
Выбранное решение
Обоснование
API Захвата
DXGI Desktop Duplication
Прямой доступ к VRAM буферу композитора, отсутствие WinRT оверхеда.2
Interop
DirectX 11 <-> CUDA
Нативная поддержка NVIDIA, возможность использования общих текстур.
Синхронизация
IDXGIKeyedMutex
Аппаратная синхронизация очередей GPU без участия CPU.23
Препроцессинг
Custom CUDA Kernel
Объединение линеаризации, смены каналов и нормализации в одну операцию.19
ML Фреймворк
LibTorch (C++)
Отсутствие GIL, прямой доступ к указателям CUDA памяти, Zero-Copy тензоры.20

Реализация данной архитектуры требует высокой квалификации в программировании графики и системном программировании, однако получаемый результат — обработка видеопотока 4K@144Гц+ в реальном времени — недостижим иными методами.
Источники
Windows 10 Screen snip DXGI screenshot - directx - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/65213270/windows-10-screen-snip-dxgi-screenshot
DXGI Desktop Duplication Screen Capture Speed [closed] - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/48278207/dxgi-desktop-duplication-screen-capture-speed
D3D11 screen desktop copy to ID3D11Texture2D - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/29661380/d3d11-screen-desktop-copy-to-id3d11texture2d
Desktop Duplication API - Win32 apps - Microsoft Learn, дата последнего обращения: февраля 5, 2026, https://learn.microsoft.com/en-us/windows/win32/direct3ddxgi/desktop-dup-api
Windows Graphics Capture vs DXGI Desktop Duplication - OBS Studio, дата последнего обращения: февраля 5, 2026, https://obsproject.com/forum/threads/windows-graphics-capture-vs-dxgi-desktop-duplication.149320/
Game capture, Window capture, or Display capture, what's the actual difference in performance? | OBS Forums - OBS Studio, дата последнего обращения: февраля 5, 2026, https://obsproject.com/forum/threads/game-capture-window-capture-or-display-capture-whats-the-actual-difference-in-performance.164599/
SerpentAI/D3DShot: Extremely fast and robust screen capture on Windows with the Desktop Duplication API - GitHub, дата последнего обращения: февраля 5, 2026, https://github.com/SerpentAI/D3DShot
Desktop duplication vs Windows Graphics capture? · Issue #24 ..., дата последнего обращения: февраля 5, 2026, https://github.com/robmikh/Win32CaptureSample/issues/24
Why obs can capture at any FPS? · obsproject obs-studio · Discussion #11486 - GitHub, дата последнего обращения: февраля 5, 2026, https://github.com/obsproject/obs-studio/discussions/11486
(OBS Recordings) Display Capture (DXGI Desktop Duplication) FIXED Frame Drops that Game Capture Introduced into my Gameplay Videos : r/obs - Reddit, дата последнего обращения: февраля 5, 2026, https://www.reddit.com/r/obs/comments/1os0z4i/obs_recordings_display_capture_dxgi_desktop/
NVIDIA CUDA Library: cudaGraphicsD3D11RegisterResource, дата последнего обращения: февраля 5, 2026, https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__D3D11_gf0545f2dd459ba49cfd6bcf2741c5ebd.html
cudaGraphicsD3D11RegisterRe, дата последнего обращения: февраля 5, 2026, https://forums.developer.nvidia.com/t/cudagraphicsd3d11registerresource-performance-any-tips/219812
How to render Direct3D scene to texture, process it with CUDA and render result to screen?, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/77237723/how-to-render-direct3d-scene-to-texture-process-it-with-cuda-and-render-result
ID3D11DeviceContext:::CopyResource is bottleneck in my particle system, дата последнего обращения: февраля 5, 2026, https://gamedev.stackexchange.com/questions/191184/id3d11devicecontextcopyresource-is-bottleneck-in-my-particle-system
D3D11 Map forces synchronization - Computer Graphics Stack Exchange, дата последнего обращения: февраля 5, 2026, https://computergraphics.stackexchange.com/questions/12978/d3d11-map-forces-synchronization
Convert ID3d11Resource to fp32 tensor in CUDA - CUDA ..., дата последнего обращения: февраля 5, 2026, https://forums.developer.nvidia.com/t/convert-id3d11resource-to-fp32-tensor-in-cuda/334133
How to convert a cudaArray to a Torch tensor? - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/77390607/how-to-convert-a-cudaarray-to-a-torch-tensor
CUDA C++ Best Practices Guide 13.1 documentation, дата последнего обращения: февраля 5, 2026, https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
NVDEC decoded frame - trying a zero copy to NV12 d3d11 texture, дата последнего обращения: февраля 5, 2026, https://forums.developer.nvidia.com/t/nvdec-decoded-frame-trying-a-zero-copy-to-nv12-d3d11-texture/123291
Libtorch memory options for tensors - pinned memory, zero copy memory - PyTorch Forums, дата последнего обращения: февраля 5, 2026, https://discuss.pytorch.org/t/libtorch-memory-options-for-tensors-pinned-memory-zero-copy-memory/157420
What's "pitch" in cudaMemcpy2DToArray and cudaMemcpy2DFromArray - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/70268991/whats-pitch-in-cudamemcpy2dtoarray-and-cudamemcpy2dfromarray
Why pytorch changes strides of tensor after inference? - Memory Format, дата последнего обращения: февраля 5, 2026, https://discuss.pytorch.org/t/why-pytorch-changes-strides-of-tensor-after-inference/122403
CUDA C++ Programming Guide (Legacy) - NVIDIA Documentation, дата последнего обращения: февраля 5, 2026, https://docs.nvidia.com/cuda/cuda-c-programming-guide/
IDXGIKeyedMutex::ReleaseSync when is rendering "done"? - Stack Overflow, дата последнего обращения: февраля 5, 2026, https://stackoverflow.com/questions/14578616/idxgikeyedmutexreleasesync-when-is-rendering-done
DirectX Synchronized Shared Surfaces & CUDA graphs - NVIDIA Developer Forums, дата последнего обращения: февраля 5, 2026, https://forums.developer.nvidia.com/t/directx-synchronized-shared-surfaces-cuda-graphs/141740


