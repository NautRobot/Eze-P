# pylint: disable=C0114,C0115,C0116
from hipfile._hipfile import driver_open, driver_close, use_count as _use_count  # pylint: disable=E0401,E0611
from hipfile.error import HipFileException


class Driver:

    @staticmethod
    def use_count():
        return _use_count()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        err = driver_close()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])

    def open(self):
        err = driver_open()
        if err[0] != 0:
            raise HipFileException(err[0], err[1])
