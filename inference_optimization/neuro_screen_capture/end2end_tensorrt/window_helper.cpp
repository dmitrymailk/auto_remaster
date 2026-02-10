#include "window_helper.h"
#include <dwmapi.h>
#include <algorithm>

#pragma comment(lib, "dwmapi.lib")

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    std::vector<WindowInfo>* windows = reinterpret_cast<std::vector<WindowInfo>*>(lParam);

    // Filter out invisible windows
    // IsWindowVisible - это функция Windows API, которая проверяет, видимо ли окно
    if (!IsWindowVisible(hwnd)) return TRUE;

    // Filter out windows with empty titles
    // GetWindowTextLength - это функция Windows API, которая возвращает длину заголовка окна
    int length = GetWindowTextLength(hwnd);
    if (length == 0) return TRUE;

    // Filter out tool windows and other non-app windows
    // GetWindowLong - это функция Windows API, которая возвращает расширенные стили окна
    // чтобы пользователю показывались только обычные приложения, а не служебные невидимые 
    // окна или мелкие панели инструментов
    LONG exStyle = GetWindowLong(hwnd, GWL_EXSTYLE);
    if (exStyle & WS_EX_TOOLWINDOW) return TRUE;
    
    // Check Cloaked state (Windows 8+) to filter out suspended UWP apps or background processes
    // DwmGetWindowAttribute - это функция Windows API, которая получает атрибуты окна
    // режим cloacked получают окна например которые находятся на другом виртуальном столе
    // нас такие не интересуют 
    int cloaked;
    HRESULT hr = DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked, sizeof(cloaked));
    if (SUCCEEDED(hr) && cloaked) return TRUE;

    // Get Title
    std::string title;
    title.resize(length + 1);
    GetWindowText(hwnd, &title[0], length + 1);
    title.resize(length); // Remove null terminator

    // Filter out Program Manager
    if (title == "Program Manager") return TRUE;

    windows->push_back({hwnd, title});
    return TRUE;
}

std::vector<WindowInfo> EnumerateWindows() {
    std::vector<WindowInfo> windows;
    // EnumWindows - это функция Windows API, которая перебирает все окна на рабочем столе
    // и вызывает для каждого окна функцию EnumWindowsProc
    // windows хранит информацию об окнах и их названиях
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));
    return windows;
}

HWND SelectWindow(const std::vector<WindowInfo>& windows) {
    if (windows.empty()) {
        std::cout << "No captureable windows found." << std::endl;
        return nullptr;
    }

    std::cout << "\nAvailable Windows:" << std::endl;
    for (size_t i = 0; i < windows.size(); ++i) {
        std::cout << "[" << i << "] " << windows[i].title << std::endl;
    }

    std::cout << "\nSelect window index (0-" << windows.size() - 1 << "): ";
    int selection;
    if (!(std::cin >> selection)) {
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        return nullptr;
    }

    if (selection >= 0 && selection < static_cast<int>(windows.size())) {
        std::cout << "Selected: " << windows[selection].title << std::endl;
        return windows[selection].hwnd;
    }

    std::cout << "Invalid selection." << std::endl;
    return nullptr;
}
