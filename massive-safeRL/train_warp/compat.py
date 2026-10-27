from functools import wraps
import os
import platform


def configure_wsl_xla():

    if 'microsoft' not in platform.release().lower():
        return False
    name = '--xla_gpu_enable_command_buffer'
    flags = [flag for flag in os.environ.get('XLA_FLAGS', '').split()
             if not flag.startswith(name + '=')]
    os.environ['XLA_FLAGS'] = ' '.join([*flags, name + '='])
    return True


def install_local_device_mapping(warp, local_devices, *, process_index):


    if warp.__version__ != '1.11.1':
        raise RuntimeError('The JAX device mapping compatibility fix requires Warp 1.11.1.')
    local_devices = tuple(local_devices)
    if any(device.platform != 'gpu' or device.process_index != process_index
           for device in local_devices):
        raise ValueError('Device mapping requires GPU devices owned by this JAX process.')
    ordinals = {int(device.id): int(device.local_hardware_id) for device in local_devices}
    mapping = (int(process_index), tuple(sorted(ordinals.items())))
    original = warp.device_from_jax
    installed = getattr(original, '_massive_saferl_local_device_mapping', None)
    if installed is not None:
        if installed != mapping:
            raise RuntimeError('The JAX device mapping changed after compatibility installation.')
        return False
    if all(global_id == ordinal for global_id, ordinal in ordinals.items()):
        return False

    @wraps(original)
    def device_from_jax(device):
        global_id = getattr(device, 'id', None)
        if (getattr(device, 'platform', None) == 'gpu'
                and getattr(device, 'process_index', None) == process_index
                and global_id in ordinals
                and getattr(device, 'local_hardware_id', None) == ordinals[global_id]):
            return warp.get_cuda_device(ordinals[global_id])
        return original(device)

    device_from_jax._massive_saferl_local_device_mapping = mapping
    warp.device_from_jax = device_from_jax
    return True
