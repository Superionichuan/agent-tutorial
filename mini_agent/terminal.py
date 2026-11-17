"""Is there a person at a keyboard? Asked before an approval question or a key prompt."""
import os
import sys


def stdin_is_terminal(stream=None) -> bool:
    """True when standard input (or the given stream) is an interactive terminal.

    On Windows, isatty() is also true for the NUL device, which is what a program gets
    when its input is empty (subprocess.DEVNULL, `< nul`, a scheduled task): a question
    asked there reads end-of-file at once. So on Windows a console handle is required too.
    """
    stream = sys.stdin if stream is None else stream
    try:
        if stream is None or not stream.isatty():
            return False
        if os.name != "nt":
            return True
        import ctypes
        import msvcrt
        handle = msvcrt.get_osfhandle(stream.fileno())
        mode = ctypes.c_uint32()
        return bool(ctypes.windll.kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
    except (AttributeError, OSError, ValueError):
        return False
