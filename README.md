EC 520 Project: Image denoising using non-LSI filters
=====================================================

Requirement
-----------

```bash
pip install numba numpy scipy scikit-image matplotlib pillow
```

Figure/table plan
-----------

Main:
- Figure 1. Example image from BSD100.
- Figure 2. Example image from Urban100.
- Figure 3 (a). Grid search over `h` for non-local means using grayscale `barbara.tif`.
- Figure 3 (b). Grid search over `h` for non-local means using RGB `bu2010.tif` (denoising in RGB space).
- Figure 3 (c). Grid search over `h` for non-local means using RGB `bu2010.tif` (denoising in LAB space). 
- Figure 4 (a). Grid search over combinations of `sigma_spatial` and `sigma_range` for bilateral filtering using grayscale `barbara.tif`.
- Figure 4 (b). Grid search over combinations of `sigma_spatial` and `sigma_range` for bilateral filtering using RGB `bu2010.tif` (denoising in RGB space).
- Figure 4 (c). Grid search over combinations of `sigma_spatial` and `sigma_range` for bilateral filtering using RGB `bu2010.tif` (denoising in LAB space).
- Figure 5. Comparison on `barbara.tif`.
- Figure 6. Comparison on `bu2010.tif`.
- Table 1. Quantitative results on BSD100 and Urban100.


Supplementary:
- Figure S1 (a). Grid search over `patch_size` and `patch_distance` for non-local means using grayscale `barbara.tif` to demonstrate insensitivity to these parameters.
- Figure S1 (b). Grid search over `patch_size` and `patch_distance` for non-local means using RGB `bu2010.tif` (denoising in RGB space) to demonstrate insensitivity to these parameters.
- Figure S1 (c). Grid search over `patch_size` and `patch_distance` for non-local means using RGB `bu2010.tif` (denoising in LAB space) to demonstrate insensitivity to these parameters.
