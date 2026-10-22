import ctypes
import json
import os
import re
import sys
import time
from pathlib import Path


MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_SESSION_BYTES = 32 * 1024 * 1024
MAX_PROCESS_SCAN_BYTES = 512 * 1024 * 1024
PROCESS_SCAN_TIMEOUT_SECONDS = 10
PROCESS_SCAN_CHUNK_BYTES = 1024 * 1024
PROCESS_SCAN_OVERLAP_BYTES = 64 * 1024

QQ_MUSIC_PROCESS_NAMES = {
    "qmbrowser.exe",
    "qqmusic.exe",
    "qqmusiccloud.exe",
    "qqmusicdesktop.exe",
    "qqmusichelper.exe",
    "qqmusicservice.exe",
}

_AUTHST_PATTERN = re.compile(
    r"""(?ix)
    (?:\\?["']?authst\\?["']?|qqmusic_key)
    \s*[:=]\s*\\?["']?
    ([a-z0-9+/_=-]{16,4096})
    """
)
_UIN_PATTERN = re.compile(
    r"""(?ix)
    (?:\\?["']?(?:uin|qqmusic_uin|musicid|str_musicid)\\?["']?)
    \s*[:=]\s*\\?["']?
    [oO]?0*(\d{4,24})
    """
)
_LOGIN_TYPE_PATTERN = re.compile(
    r"""(?ix)
    (?:\\?["']?(?:tmeLoginType|loginType|login_type)\\?["']?)
    \s*[:=]\s*\\?["']?
    ([123])
    """
)


def _normalized_uin(value: str) -> str | None:
    normalized = re.sub(r"^[oO]", "", value.strip())
    normalized = re.sub(r"^0+(?=\d{4,24}$)", "", normalized)
    return normalized if re.fullmatch(r"\d{4,24}", normalized) else None


def _decode_config(data: bytes) -> str:
    if data.startswith(b"\xff\xfe"):
        return data[2:].decode("utf-16le", errors="replace")
    if data.startswith(b"\xfe\xff"):
        return data[2:].decode("utf-16be", errors="replace")
    return data.removeprefix(b"\xef\xbb\xbf").decode(
        "utf-8",
        errors="replace",
    )


def parse_qqmusic_windows_uin(data: bytes) -> str | None:
    text = _decode_config(data)
    section = ""
    fallback = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue
        match = re.match(r"(?i)^uin\s*=\s*['\"]?([^'\"\s]+)", line)
        if not match:
            continue
        candidate = _normalized_uin(match.group(1))
        if not candidate:
            continue
        if section == "account":
            return candidate
        if fallback is None:
            fallback = candidate
    return fallback


def _decoded_session_texts(data: bytes) -> list[str]:
    decoded = {
        data.decode("utf-8", errors="replace"),
        data.decode("latin1", errors="replace"),
    }
    if len(data) >= 2:
        even = data[: len(data) - (len(data) % 2)]
        decoded.add(even.decode("utf-16le", errors="replace"))
        if len(data) > 2:
            shifted = data[1 : len(data) - ((len(data) - 1) % 2)]
            decoded.add(shifted.decode("utf-16le", errors="replace"))
    return [text for text in decoded if text]


def extract_qqmusic_windows_credential(
    data: bytes,
    current_uin: str,
) -> dict[str, object] | None:
    normalized_current = _normalized_uin(current_uin)
    if not normalized_current:
        return None
    for text in _decoded_session_texts(data):
        for auth_match in _AUTHST_PATTERN.finditer(text):
            start = max(0, auth_match.start() - 16_384)
            end = min(len(text), auth_match.end() + 16_384)
            record = text[start:end]
            record_uins = {
                candidate
                for match in _UIN_PATTERN.finditer(record)
                if (candidate := _normalized_uin(match.group(1)))
            }
            if normalized_current not in record_uins:
                continue
            music_key = auth_match.group(1)
            login_match = _LOGIN_TYPE_PATTERN.search(record)
            login_type = (
                int(login_match.group(1))
                if login_match
                else 1 if music_key.startswith("W_X") else 3
            )
            return {
                "musicId": normalized_current,
                "musicKey": music_key,
                "loginType": login_type,
            }
    return None


def _qqmusic_root() -> Path:
    app_data = os.environ.get("APPDATA", "").strip()
    if not app_data:
        raise RuntimeError("无法定位 QQ 音乐配置目录")
    return Path(app_data) / "Tencent" / "QQMusic"


def _bounded_read(path: Path, limit: int) -> bytes | None:
    try:
        stats = path.stat()
        if not path.is_file() or stats.st_size <= 0 or stats.st_size > limit:
            return None
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        return data if len(data) <= limit else None
    except OSError:
        return None


def _current_qqmusic_uin(root: Path) -> str:
    for name in ("QQMusicServiceConfig.ini", "QQMusicConfig.ini"):
        data = _bounded_read(root / name, MAX_CONFIG_BYTES)
        if data and (uin := parse_qqmusic_windows_uin(data)):
            return uin
    raise RuntimeError(
        "未读取到 QQ 音乐当前账号；请保持 QQ 音乐客户端运行并完成登录"
    )


def _credential_from_session_files(
    root: Path,
    current_uin: str,
) -> dict[str, object] | None:
    for relative in (
        "SetCookie.dat",
        "_SetCookie.dat",
        os.path.join("mmkv", "mmkv.default"),
    ):
        data = _bounded_read(root / relative, MAX_SESSION_BYTES)
        if not data:
            continue
        credential = extract_qqmusic_windows_credential(data, current_uin)
        if credential:
            return credential
    return None


def _process_entry_type():
    from ctypes import wintypes

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("size", wintypes.DWORD),
            ("usage", wintypes.DWORD),
            ("process_id", wintypes.DWORD),
            ("default_heap_id", ctypes.c_size_t),
            ("module_id", wintypes.DWORD),
            ("threads", wintypes.DWORD),
            ("parent_process_id", wintypes.DWORD),
            ("priority_class_base", wintypes.LONG),
            ("flags", wintypes.DWORD),
            ("executable", wintypes.WCHAR * 260),
        ]

    return ProcessEntry32W


def _memory_basic_information_type():
    from ctypes import wintypes

    class MemoryBasicInformation(ctypes.Structure):
        _fields_ = [
            ("base_address", ctypes.c_void_p),
            ("allocation_base", ctypes.c_void_p),
            ("allocation_protect", wintypes.DWORD),
            ("partition_id", wintypes.WORD),
            ("region_size", ctypes.c_size_t),
            ("state", wintypes.DWORD),
            ("protect", wintypes.DWORD),
            ("kind", wintypes.DWORD),
        ]

    return MemoryBasicInformation


def _same_windows_user(kernel32, advapi32, process_handle) -> bool:
    from ctypes import wintypes

    token_query = 0x0008
    token_user_class = 1
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [
            ("sid", ctypes.c_void_p),
            ("attributes", wintypes.DWORD),
        ]

    class TokenUser(ctypes.Structure):
        _fields_ = [("user", SidAndAttributes)]

    def token_user(handle):
        token = wintypes.HANDLE()
        if not advapi32.OpenProcessToken(
            handle,
            token_query,
            ctypes.byref(token),
        ):
            return None
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(
            token,
            token_user_class,
            None,
            0,
            ctypes.byref(size),
        )
        if size.value <= 0 or size.value > 64 * 1024:
            kernel32.CloseHandle(token)
            return None
        buffer = ctypes.create_string_buffer(size.value)
        ok = advapi32.GetTokenInformation(
            token,
            token_user_class,
            buffer,
            size,
            ctypes.byref(size),
        )
        kernel32.CloseHandle(token)
        if not ok:
            return None
        sid = ctypes.cast(
            buffer,
            ctypes.POINTER(TokenUser),
        ).contents.user.sid
        return buffer, sid

    current = token_user(kernel32.GetCurrentProcess())
    target = token_user(process_handle)
    if not current or not target:
        return False
    return bool(advapi32.EqualSid(current[1], target[1]))


def _verified_process_image(kernel32, process_handle, expected_name: str) -> bool:
    from ctypes import wintypes

    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    size = wintypes.DWORD(32_768)
    buffer = ctypes.create_unicode_buffer(size.value)
    if not kernel32.QueryFullProcessImageNameW(
        process_handle,
        0,
        buffer,
        ctypes.byref(size),
    ):
        return False
    path = Path(buffer.value)
    if path.name.lower() != expected_name:
        return False
    return "qqmusic" in {
        part.lower()
        for part in path.parts
    }


def _qqmusic_processes(kernel32) -> list[tuple[int, str]]:
    from ctypes import wintypes

    th32cs_snapprocess = 0x00000002
    invalid_handle_value = ctypes.c_void_p(-1).value
    ProcessEntry32W = _process_entry_type()

    kernel32.CreateToolhelp32Snapshot.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32W),
    ]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(th32cs_snapprocess, 0)
    if snapshot == invalid_handle_value:
        return []
    processes = []
    entry = ProcessEntry32W()
    entry.size = ctypes.sizeof(entry)
    try:
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            name = entry.executable.lower()
            if name in QQ_MUSIC_PROCESS_NAMES:
                processes.append((int(entry.process_id), name))
            entry.size = ctypes.sizeof(entry)
            has_entry = kernel32.Process32NextW(
                snapshot,
                ctypes.byref(entry),
            )
    finally:
        kernel32.CloseHandle(snapshot)
    return processes


def _scan_process(
    kernel32,
    advapi32,
    process_id: int,
    expected_name: str,
    current_uin: str,
    deadline: float,
    budget: int,
) -> tuple[dict[str, object] | None, int]:
    from ctypes import wintypes

    process_query_information = 0x0400
    process_vm_read = 0x0010
    mem_commit = 0x1000
    mem_private = 0x20000
    mem_image = 0x1000000
    page_guard = 0x100
    readable_pages = {0x02, 0x04, 0x08, 0x20, 0x40, 0x80}
    MemoryBasicInformation = _memory_basic_information_type()

    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.VirtualQueryEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(MemoryBasicInformation),
        ctypes.c_size_t,
    ]
    kernel32.VirtualQueryEx.restype = ctypes.c_size_t
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    process = kernel32.OpenProcess(
        process_query_information | process_vm_read,
        False,
        process_id,
    )
    if not process:
        return None, 0
    scanned = 0
    try:
        if not _verified_process_image(kernel32, process, expected_name):
            return None, scanned
        if not _same_windows_user(kernel32, advapi32, process):
            return None, scanned
        address = 0
        while scanned < budget and time.monotonic() < deadline:
            region = MemoryBasicInformation()
            result = kernel32.VirtualQueryEx(
                process,
                ctypes.c_void_p(address),
                ctypes.byref(region),
                ctypes.sizeof(region),
            )
            if not result or not region.region_size:
                break
            base = int(region.base_address or 0)
            next_address = base + int(region.region_size)
            if next_address <= address:
                break
            protection = int(region.protect) & 0xFF
            if (
                region.state == mem_commit
                and not (region.protect & page_guard)
                and protection in readable_pages
                and region.kind in {mem_private, mem_image}
            ):
                position = base
                end = min(next_address, base + budget - scanned)
                carry = b""
                while (
                    position < end
                    and scanned < budget
                    and time.monotonic() < deadline
                ):
                    size = min(PROCESS_SCAN_CHUNK_BYTES, end - position)
                    buffer = ctypes.create_string_buffer(size)
                    bytes_read = ctypes.c_size_t()
                    kernel32.ReadProcessMemory(
                        process,
                        ctypes.c_void_p(position),
                        buffer,
                        size,
                        ctypes.byref(bytes_read),
                    )
                    scanned += size
                    if bytes_read.value:
                        window = carry + buffer.raw[: bytes_read.value]
                        credential = extract_qqmusic_windows_credential(
                            window,
                            current_uin,
                        )
                        ctypes.memset(buffer, 0, size)
                        if credential:
                            return credential, scanned
                        carry = window[-PROCESS_SCAN_OVERLAP_BYTES:]
                    else:
                        ctypes.memset(buffer, 0, size)
                    position += size
            address = next_address
    finally:
        kernel32.CloseHandle(process)
    return None, scanned


def _credential_from_running_client(
    current_uin: str,
) -> dict[str, object] | None:
    if sys.platform != "win32":
        return None

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    deadline = time.monotonic() + PROCESS_SCAN_TIMEOUT_SECONDS
    remaining = MAX_PROCESS_SCAN_BYTES
    for process_id, name in _qqmusic_processes(kernel32):
        if remaining <= 0 or time.monotonic() >= deadline:
            break
        credential, scanned = _scan_process(
            kernel32,
            advapi32,
            process_id,
            name,
            current_uin,
            deadline,
            remaining,
        )
        remaining -= scanned
        if credential:
            return credential
    return None


def verify_qqmusic_windows_runtime() -> dict[str, object]:
    if sys.platform != "win32":
        raise RuntimeError("QQ 音乐 Windows runtime self-test 仅支持 Windows")
    process_entry_bytes = ctypes.sizeof(_process_entry_type())
    memory_information_bytes = ctypes.sizeof(
        _memory_basic_information_type()
    )
    if process_entry_bytes != 568 or memory_information_bytes != 48:
        raise RuntimeError(
            "Windows QQ 音乐运行时结构不兼容："
            f"PROCESSENTRY32W={process_entry_bytes}, "
            f"MEMORY_BASIC_INFORMATION={memory_information_bytes}"
        )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    for library, names in (
        (
            kernel32,
            (
                "CreateToolhelp32Snapshot",
                "OpenProcess",
                "Process32FirstW",
                "Process32NextW",
                "QueryFullProcessImageNameW",
                "ReadProcessMemory",
                "VirtualQueryEx",
            ),
        ),
        (
            advapi32,
            (
                "EqualSid",
                "GetTokenInformation",
                "OpenProcessToken",
            ),
        ),
    ):
        for name in names:
            getattr(library, name)
    return {
        "platform": "win32",
        "processEntryBytes": process_entry_bytes,
        "memoryInformationBytes": memory_information_bytes,
    }


def load_qqmusic_windows_credential() -> dict[str, object]:
    if sys.platform != "win32":
        raise RuntimeError("QQ 音乐客户端登录态读取仅支持 Windows")
    root = _qqmusic_root()
    current_uin = _current_qqmusic_uin(root)
    credential = _credential_from_session_files(root, current_uin)
    if credential:
        return credential
    credential = _credential_from_running_client(current_uin)
    if credential:
        return credential
    raise RuntimeError(
        "已读取 QQ 音乐当前账号，但未取得可用会话令牌。"
        "请保持 QQ 音乐客户端运行并已登录，"
        "且不要让两个应用使用不同的 Windows 用户或管理员权限。"
    )


def print_qqmusic_windows_credential() -> None:
    print(
        json.dumps(
            load_qqmusic_windows_credential(),
            ensure_ascii=False,
        ),
        flush=True,
    )


def print_qqmusic_windows_runtime_self_test() -> None:
    print(
        json.dumps(
            verify_qqmusic_windows_runtime(),
            ensure_ascii=False,
        ),
        flush=True,
    )
