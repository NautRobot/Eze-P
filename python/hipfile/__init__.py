from hipfile._hipfile import *  # noqa: F401,F403
from hipfile._hipfile import (
    # Constants
    VERSION_MAJOR,
    VERSION_MINOR,
    VERSION_PATCH,
    BASE_ERR,
    # Error Macros
    is_hipfile_err,
    hipfile_errstr,
    is_hip_drv_err,
    hip_drv_err,
)
from hipfile.buffer import(
    Buffer
)
from hipfile.driver import (
    Driver
)
from hipfile.error import (
    HipFileException
)
from hipfile.file import (
    FileHandle
)
from hipfile.properties import (
    driver_get_properties,
    get_version
)

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"
