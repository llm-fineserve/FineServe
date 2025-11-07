# shm_utils.py
import os
from multiprocessing.shared_memory import SharedMemory
from fineserve.utils.workload_utils import Request
from typing import List, Union
import pickle
import copy
from fineserve.utils.constant import (SM_HOLD_NAME_FMT, OUT_HOLD_NAME_FMT, ADD_REQ_NAME_FMT,
                                      RET_REQ_NAME_FMT, PREEMPT_REQ_NAME_FMT)


class SharedMemoryManager:
    """A class to manage shared memory operations for FineServe."""

    def __init__(self):
       pass

    @staticmethod
    def map_shm_name(shm_name: str) -> str:
        """Map shared memory name with prefix."""
        shm_name = shm_name.split("/")[-1]
        return shm_name

    @staticmethod
    def create_shared_var(shm_name: str,
                          size: int = 0,
                          create: bool = True) -> SharedMemory:
        """Create or recreate a shared memory segment."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        # first close and unlink the shared memory if it exists and create is True
        if create:
            try:
                shm = SharedMemory(shm_name)
                shm.close()
                shm.unlink()
            except:
                pass

        shm = SharedMemory(shm_name, create=create, size=size)
        if create:
            shm.buf[:] = b"0" * size
        return shm

    @staticmethod
    def read_shared_var(shm: SharedMemory) -> int:
        """Read integer value from shared memory."""
        data = int(bytes(shm.buf[:]).decode('utf-8'))
        return data

    @staticmethod
    def write_shared_var(shm: SharedMemory,
                         data: int,
                         close: bool = False) -> None:
        """Write integer value to shared memory."""
        sign = "-" if data < 0 else "+"
        data = str(abs(data))
        data = sign + "0" * (shm.size - len(data) - 1) + data
        shm.buf[:] = data.encode('utf-8')
        if close:
            shm.close()

    @staticmethod
    def dump_reqs_to_shared_var(shm_name: str, data: List[Request]) -> None:
        """Dump list of Request objects to shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)

        # Serialize the list of Request objects
        serialized_data = pickle.dumps(copy.deepcopy(data))

        shm = SharedMemory(shm_name, create=True, size=len(serialized_data))
        # Write the serialized data to the shared memory
        shm.buf[:len(serialized_data)] = serialized_data
        shm.close()

    @staticmethod
    def load_reqs_from_shared_var(shm_name: str) -> List[Union[Request, int]]:
        """Load list of Request objects from shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        try:
            shm = SharedMemory(shm_name)
        except:
            data = []
            return data

        # Read the serialized data from the shared memory
        serialized_data = bytes(shm.buf[:])

        # Deserialize the data back into a list of Request objects
        data = pickle.loads(serialized_data)

        shm.close()
        shm.unlink()

        return data

    @staticmethod
    def dump_to_shared_var(shm_name: str, data: List[int]) -> None:
        """Dump list of integers to shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        data = ",".join([str(x) for x in data])
        shm = SharedMemory(shm_name, create=True, size=len(data))
        shm.buf[:] = data.encode('utf-8')
        shm.close()

    @staticmethod
    def load_from_shared_var(shm_name: str) -> List[int]:
        """Load list of integers from shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        try:
            shm = SharedMemory(shm_name)
        except:
            data = []
            return data
        data = bytes(shm.buf[:]).decode('utf-8')
        while "\x00" in data:
            data = bytes(shm.buf[:]).decode('utf-8')
        data = [int(x) for x in data.split(",")]
        shm.close()
        shm.unlink()
        return data

    @staticmethod
    def write_str_to_shared_var(shm_name: str, data: str) -> None:
        """Write string to shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        shm = SharedMemory(shm_name, create=True, size=len(data))
        shm.buf[:] = data.encode('utf-8')
        shm.close()

    @staticmethod
    def read_str_from_shared_var(shm_name: str) -> str:
        """Read string from shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        try:
            shm = SharedMemory(shm_name)
        except:
            data = ""
            return data
        data = bytes(shm.buf[:]).decode('utf-8')
        while "\x00" in data:
            data = bytes(shm.buf[:]).decode('utf-8')
        shm.close()
        return data

    @staticmethod
    def write_list_to_shared_var(shm_name: str, data: List[int]) -> None:
        """Write list of integers to shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        data = ",".join([str(x) for x in data])
        shm = SharedMemory(shm_name, create=True, size=len(data))
        shm.buf[:] = data.encode('utf-8')
        shm.close()

    @staticmethod
    def read_list_from_shared_var(shm_name: str) -> List[int]:
        """Read list of integers from shared memory."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        try:
            shm = SharedMemory(shm_name)
        except:
            data = []
            return data
        data = bytes(shm.buf[:]).decode('utf-8')
        while "\x00" in data:
            data = bytes(shm.buf[:]).decode('utf-8')
        data = [int(x) for x in data.split(",")]
        shm.close()
        shm.unlink()
        return data

    @staticmethod
    def close_shared_var(shm_name: str) -> None:
        """Close and unlink shared memory segment."""
        shm_name = SharedMemoryManager.map_shm_name(shm_name)
        try:
            shm = SharedMemory(shm_name)
            shm.close()
            shm.unlink()
        except:
            pass

    @staticmethod
    def get_string_format_options() -> dict:
        """Get string format options used for shared memory names."""
        return {
            'sm_hold': SM_HOLD_NAME_FMT,
            'out_hold': OUT_HOLD_NAME_FMT,
            'add_req': ADD_REQ_NAME_FMT,
            'ret_req': RET_REQ_NAME_FMT,
            'preempt_req': PREEMPT_REQ_NAME_FMT
        }

    @staticmethod
    def get_shared_memory_name(model_name: str, mps_percentage: int, format_type: str) -> str:
        """
        Generate shared memory name based on format type.

        Args:
            model_name: Name of the model
            mps_percentage: MPS percentage
            format_type: Type of format ('sm_hold', 'out_hold', 'add_req', 'ret_req', 'preempt_req')

        Returns:
            Formatted shared memory name
        """
        formats = SharedMemoryManager.get_string_format_options()
        if format_type not in formats:
            raise ValueError(f"Unknown format type: {format_type}")

        return formats[format_type].format(model_name, mps_percentage)

    @staticmethod
    def cleanup_shared_memory():
        import os
        import glob
        all_fmts = [
            SM_HOLD_NAME_FMT,
            OUT_HOLD_NAME_FMT,
            ADD_REQ_NAME_FMT,
            RET_REQ_NAME_FMT,
            PREEMPT_REQ_NAME_FMT,
        ]

        fmts = [f"_{fmt.split('_')[1]}_" for fmt in all_fmts]
        files = glob.glob("/dev/shm/*")
        for f in files:
            if not isinstance(f, str):
                continue
            if f.startswith('/dev/shm/eic-'):
                continue    # AWS-specific
            for fmt in fmts:
                if fmt in f:    # for safety
                    os.remove(f)
