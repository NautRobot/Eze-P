from hipfile._hipfile import (
    buf_deregister as _buf_deregister,
    buf_register as _buf_register
)
from hipfile.error import HipFileException

def buf_register(device_buffer, length, flags):
    err = _buf_register(device_buffer, length, flags)
    if (err[0] != 0):
        raise HipFileException(err[0], err[1])

def buf_deregister(device_buffer):
    err = _buf_deregister(device_buffer)
    if (err[0] != 0):
        raise HipFileException(err[0], err[1])
