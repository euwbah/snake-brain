import ctypes
import sys
from decimal import Context, setcontext
from time import sleep
from models.cubic_estimation import train_model, test_model

REGULAR_DECIMAL_CONTEXT = Context(prec=28)
setcontext(REGULAR_DECIMAL_CONTEXT)

if __name__ == "__main__":

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 0b111)

    # d = generate_data(10)

    test_model('fx=x2+x+1 1-400-0.5 decay0', 1, 400, 0.5)

    train_model('fx=x2+x+1 1-400-0.5 decay0',
                1, 400, 0.5, 100, 1,
                0, 1, 1, 1,
                -6, 6,
                -8, 8,
                60000, 4000, 300, 75,
                step_size=0.00005, momentum=0.8, decay=0.000003,
                pause_after_iter=None)

    exit(0)
