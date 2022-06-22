from typing import List


def flatten(l: List[List]) -> List:
    return [item for sublist in l for item in sublist]
