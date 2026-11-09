"""R's default random number generator (Mersenne-Twister, ``sample.kind = "Rejection"``).

Only used to reproduce mgcv's random knot selection for thin plate splines with
more than ``max.knots`` unique covariate values (``set.seed(1); sample(...)``).
"""

from __future__ import annotations

import math

_N, _M = 624, 397
_MATRIX_A, _UPPER, _LOWER = 0x9908B0DF, 0x80000000, 0x7FFFFFFF
_MASK32 = 0xFFFFFFFF
_I2_32M1 = 2.328306437080797e-10


class RRandom:
    def __init__(self, seed: int):
        s = int(seed) & _MASK32
        for _ in range(50):
            s = (69069 * s + 1) & _MASK32
        state = []
        for _ in range(_N + 1):
            s = (69069 * s + 1) & _MASK32
            state.append(s)
        self.mt = state[1:]
        self.mti = _N  # FixupSeeds: dummy[0] = 624

    def _genrand(self) -> float:
        mt = self.mt
        if self.mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self.mti = 0
        y = mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return (y & _MASK32) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        v = self._genrand()
        if v <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - v <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return v

    def _rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv


def r_sample(n: int, k: int, seed: int):
    """``set.seed(seed); sample.int(n, k)`` (1-based indices)."""
    rng = RRandom(seed)
    x = list(range(n))
    out = []
    nn = n
    for _ in range(k):
        j = int(rng.unif_index(nn))
        out.append(x[j] + 1)
        nn -= 1
        x[j] = x[nn]
    return out
