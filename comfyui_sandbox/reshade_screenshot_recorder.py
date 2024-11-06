import pyautogui
from pynput import keyboard
import time

controller = keyboard.Controller()

is_started = False


def press_screen():
    global is_started
    while is_started:
        controller.press(keyboard.Key.print_screen)
        time.sleep(0.1)
        controller.release(keyboard.Key.print_screen)


def on_press(key):
    # try:
    #     print("alphanumeric key {0} pressed".format(key.char))
    # except AttributeError:
    #     print("special key {0} pressed".format(key))
    if key == keyboard.Key.f8:
        global is_started
        is_started = True
        press_screen()


def on_release(key):
    print("{0} released".format(key))
    if key == keyboard.Key.f9:
        # Stop listener
        global is_started
        is_started = False
        print("Stop listener")
        return False


with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
