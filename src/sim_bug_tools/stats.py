import pandas as pd

def rectangle_wave(time : list[int], is_bug : list[bool]):
    assert len(time) == len(is_bug)
    return