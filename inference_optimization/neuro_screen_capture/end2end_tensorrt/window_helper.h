#pragma once

#include <windows.h>
#include <string>
#include <vector>
#include <iostream>

struct WindowInfo {
    HWND hwnd;
    std::string title;
};

// Returns a list of visible application windows
std::vector<WindowInfo> EnumerateWindows();

// Prints the list to console and asks user to select one by index
// Returns the selected HWND, or nullptr if canceled/invalid
HWND SelectWindow(const std::vector<WindowInfo>& windows);
