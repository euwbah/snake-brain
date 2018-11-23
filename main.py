import ctypes
import sys
from time import sleep
from models.cubic_estimation import train_model, test_model

if __name__ == "__main__":

    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 0b111)

    # d = generate_data(10)

    # test_model('fx=x2+x+1 1-20-0.5 decay0', 1, 20, 0.5)

    train_model('fx=x2+x+1 3-3-0.5 decay0',
                3, 3, 0.5, 200, 10,
                0, 1, 1, 1,
                -6, 6,
                -10, 10,
                60000, 4000, 300, 75,
                step_size=0.001, momentum=0.5, decay=0.00001)

    exit(0)
