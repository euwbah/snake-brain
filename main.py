import ctypes
import sys
from decimal import Context, setcontext
from time import sleep
from models.cubic_estimation import train_model, test_model

REGULAR_DECIMAL_CONTEXT = Context(prec=28, Emax=10_000_000, Emin=-10_000_000)
setcontext(REGULAR_DECIMAL_CONTEXT)

if __name__ == "__main__":

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 0b111)

    # d = generate_data(10)

    test_model('fx=x2+1 2-30-0.5 decay3e-6', 2, 30, 0.5, 226)

    train_model('fx=x2+1 2-30-0.5 decay3e-6',
                2, 30, 0.5, 240, 1,
                0, 1, 0, 1,
                -6, 6,
                -8, 8,
                60000, 4000, 300, 75,
                step_size=0.00005, momentum=0.1, decay=0,
                log_level=0,
                pause_after_iter=None)

    exit(0)
