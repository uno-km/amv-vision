import json
import struct
import numpy as np

def save_safetensors(tensors_dict: dict, file_path: str):
    """
    Save dictionary of NumPy arrays to SafeTensors format with zero PyTorch dependency.
    """
    header = {}
    offset = 0
    buffer_bytes = bytearray()

    dtype_map = {
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("uint8"): "U8",
    }

    for name, arr in tensors_dict.items():
        arr_cont = np.ascontiguousarray(arr)
        raw = arr_cont.tobytes()
        length = len(raw)
        st_dtype = dtype_map.get(arr_cont.dtype, "F32")
        
        header[name] = {
            "dtype": st_dtype,
            "shape": list(arr_cont.shape),
            "data_offsets": [offset, offset + length]
        }
        offset += length
        buffer_bytes.extend(raw)

    header_json = json.dumps(header).encode("utf-8")
    header_len = len(header_json)

    with open(file_path, "wb") as f:
        # 8-byte little endian header length
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        f.write(buffer_bytes)

MAX_SAFETENSORS_HEADER_BYTES = 32 * 1024 * 1024  # 32 MB maximum header

def load_safetensors(file_path: str) -> dict:
    """
    Load SafeTensors file into dictionary of NumPy arrays with strict security validation.
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SafeTensors file not found: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size < 8:
        raise ValueError(f"Invalid SafeTensors file: size ({file_size} bytes) is less than 8-byte header length.")

    dtype_inv_map = {
        "F32": np.float32,
        "F64": np.float64,
        "I32": np.int32,
        "I64": np.int64,
        "U8": np.uint8,
    }

    with open(file_path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        
        if header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"SafeTensors header length ({header_len} bytes) exceeds maximum safe limit ({MAX_SAFETENSORS_HEADER_BYTES} bytes).")
        if 8 + header_len > file_size:
            raise ValueError(f"SafeTensors header length ({header_len}) exceeds total file size ({file_size}).")

        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode("utf-8"))

        data_start = 8 + header_len
        tensors = {}

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            np_dtype = dtype_inv_map.get(meta.get("dtype"), np.float32)
            shape = meta.get("shape", [])
            data_offsets = meta.get("data_offsets", [0, 0])
            start_off, end_off = data_offsets[0], data_offsets[1]
            
            if start_off < 0 or end_off < start_off or (data_start + end_off) > file_size:
                raise ValueError(f"Invalid tensor offsets for '{name}': [{start_off}, {end_off}] outside file bounds ({file_size}).")

            f.seek(data_start + start_off)
            raw = f.read(end_off - start_off)
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            tensors[name] = arr.copy()

    return tensors
