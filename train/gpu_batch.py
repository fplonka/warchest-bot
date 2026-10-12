from functools import lru_cache

import numpy as np
import torch

import mirror
import warchest


@lru_cache(maxsize=None)
def _tables(device_text):
    device = torch.device(device_text)
    cards = torch.as_tensor(np.asarray(warchest.card_features_table(), np.float32), device=device)
    locations = torch.as_tensor(np.asarray(warchest.hex_location_flags(), np.uint8), device=device)
    return cards, locations


def expand(rows, device):
    rows = torch.as_tensor(np.ascontiguousarray(rows)).pin_memory().to(device, non_blocking=True)
    cards, locations = _tables(str(device))
    x = torch.empty((len(rows), warchest.PUBFEAT), dtype=torch.float32, device=device)
    ordinal = device.index if device.index is not None else torch.cuda.current_device()
    warchest.expand_rows_cuda(rows.data_ptr(), cards.data_ptr(), locations.data_ptr(), x.data_ptr(),
                              len(rows), torch.cuda.current_stream(device).cuda_stream, ordinal)
    return x


def make_batch(parts, device):
    rows, desc, parow, target, z = parts
    rows, desc = mirror.to_mover(rows, desc, parow)
    t = lambda a, dtype: torch.as_tensor(a, dtype=dtype).pin_memory().to(device, non_blocking=True)
    return (expand(rows, device), t(desc, torch.uint8), t(parow, torch.long),
            t(target, torch.float32), t(z, torch.float32))
