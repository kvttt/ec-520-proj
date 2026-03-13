EC 520 Project: Image denoising using non-LSI filters
=====================================================

Currently, we have implemented the non-local means (NLM) algorithm that is more than 10x faster than the implementation in scikit-image. The following results is obtained on a 512x512 image on an Apple M4 Max CPU with 36 GB memory.

```text
skimage: 4818.30 ms
nlm_numpy: 4460.29 ms
nlm_numba: 460.19 ms
nlm_numpy vs skimage: RMSE = 2.30e-03
nlm_numba vs skimage: RMSE = 2.22e-03
```

Requirement
-----------

```bash
pip install numba numpy scipy scikit-image
```

Usage
-----

See `nlm_benchmark.py` for the benchmark code. 
