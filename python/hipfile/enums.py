from enum import IntEnum

from hipfile._hipfile import (
    # hipFileOpError_t values (resolved from C at build time)
    py_hipFileSuccess,
    py_hipFileDriverNotInitialized,
    py_hipFileDriverInvalidProps,
    py_hipFileDriverUnsupportedLimit,
    py_hipFileDriverVersionMismatch,
    py_hipFileDriverVersionReadError,
    py_hipFileDriverClosing,
    py_hipFilePlatformNotSupported,
    py_hipFileIONotSupported,
    py_hipFileDeviceNotSupported,
    py_hipFileDriverError,
    py_hipFileHipDriverError,
    py_hipFileHipPointerInvalid,
    py_hipFileHipMemoryTypeInvalid,
    py_hipFileHipPointerRangeError,
    py_hipFileHipContextMismatch,
    py_hipFileInvalidMappingSize,
    py_hipFileInvalidMappingRange,
    py_hipFileInvalidFileType,
    py_hipFileInvalidFileOpenFlag,
    py_hipFileDIONotSet,
    py_hipFileInvalidValue,
    py_hipFileMemoryAlreadyRegistered,
    py_hipFileMemoryNotRegistered,
    py_hipFilePermissionDenied,
    py_hipFileDriverAlreadyOpen,
    py_hipFileHandleNotRegistered,
    py_hipFileHandleAlreadyRegistered,
    py_hipFileDeviceNotFound,
    py_hipFileInternalError,
    py_hipFileGetNewFDFailed,
    py_hipFileDriverSetupError,
    py_hipFileIODisabled,
    py_hipFileBatchSubmitFailed,
    py_hipFileGPUMemoryPinningFailed,
    py_hipFileBatchFull,
    py_hipFileAsyncNotSupported,
    py_hipFileIOMaxError,
    # hipFileFileHandleType_t values (resolved from C at build time)
    py_hipFileHandleTypeOpaqueFD,
    py_hipFileHandleTypeOpaqueWin32,
    py_hipFileHandleTypeUserspaceFS,
)


class OpError(IntEnum):
    """Python enum mirroring hipFileOpError_t.

    Values are sourced from the C enum via the Cython layer, not
    redefined.  Rebuilding the extension picks up any value changes
    in hipfile.h automatically.
    """

    Success                 = py_hipFileSuccess
    DriverNotInitialized    = py_hipFileDriverNotInitialized
    DriverInvalidProps      = py_hipFileDriverInvalidProps
    DriverUnsupportedLimit  = py_hipFileDriverUnsupportedLimit
    DriverVersionMismatch   = py_hipFileDriverVersionMismatch
    DriverVersionReadError  = py_hipFileDriverVersionReadError
    DriverClosing           = py_hipFileDriverClosing
    PlatformNotSupported    = py_hipFilePlatformNotSupported
    IONotSupported          = py_hipFileIONotSupported
    DeviceNotSupported      = py_hipFileDeviceNotSupported
    DriverError             = py_hipFileDriverError
    HipDriverError          = py_hipFileHipDriverError
    HipPointerInvalid       = py_hipFileHipPointerInvalid
    HipMemoryTypeInvalid    = py_hipFileHipMemoryTypeInvalid
    HipPointerRangeError    = py_hipFileHipPointerRangeError
    HipContextMismatch      = py_hipFileHipContextMismatch
    InvalidMappingSize      = py_hipFileInvalidMappingSize
    InvalidMappingRange     = py_hipFileInvalidMappingRange
    InvalidFileType         = py_hipFileInvalidFileType
    InvalidFileOpenFlag     = py_hipFileInvalidFileOpenFlag
    DIONotSet               = py_hipFileDIONotSet
    InvalidValue            = py_hipFileInvalidValue
    MemoryAlreadyRegistered = py_hipFileMemoryAlreadyRegistered
    MemoryNotRegistered     = py_hipFileMemoryNotRegistered
    PermissionDenied        = py_hipFilePermissionDenied
    DriverAlreadyOpen       = py_hipFileDriverAlreadyOpen
    HandleNotRegistered     = py_hipFileHandleNotRegistered
    HandleAlreadyRegistered = py_hipFileHandleAlreadyRegistered
    DeviceNotFound          = py_hipFileDeviceNotFound
    InternalError           = py_hipFileInternalError
    GetNewFDFailed          = py_hipFileGetNewFDFailed
    DriverSetupError        = py_hipFileDriverSetupError
    IODisabled              = py_hipFileIODisabled
    BatchSubmitFailed       = py_hipFileBatchSubmitFailed
    GPUMemoryPinningFailed  = py_hipFileGPUMemoryPinningFailed
    BatchFull               = py_hipFileBatchFull
    AsyncNotSupported       = py_hipFileAsyncNotSupported
    IOMaxError              = py_hipFileIOMaxError


class FileHandleType(IntEnum):
    """Python enum mirroring hipFileFileHandleType_t.

    Values are sourced from the C enum via the Cython layer.
    """

    OpaqueFD    = py_hipFileHandleTypeOpaqueFD
    OpaqueWin32 = py_hipFileHandleTypeOpaqueWin32
    UserspaceFS = py_hipFileHandleTypeUserspaceFS
