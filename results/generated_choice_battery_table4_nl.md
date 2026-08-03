# Generated Choice Benchmark Battery (table4_nl)

All cross-estimator runtimes report estimation plus covariance on one logical CPU. Synthetic cases vary sample size (N), number of alternatives (J), number of observed variables (K), and equicorrelation (rho). MNL rows compare TorchDCM against Torch-Choice, SciPy, Biogeme, Apollo, mlogit, gmnl, and xlogit where available. Nested-logit rows also include Torch-Choice, while mixed-logit rows compare TorchDCM against Biogeme and Apollo. A dagger marks a completed solver whose final log likelihood is clearly below the row best; its runtime is retained but its estimate is excluded from consistency.

| case | family | N | J | K | rho | TorchDCM | Torch-Choice | SciPy | Biogeme | Apollo | mlogit | gmnl | xlogit | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| nl_N_1000 | Nested logit | 1000 | 4 | None | None | 0.046 | 0.059 | NA | 24.958 | 0.519 | NA | NA | NA | Yes |
| nl_N_10000 | Nested logit | 10000 | 4 | None | None | 0.070 | 0.143 | NA | 25.894 | 2.052 | NA | NA | NA | Yes |
| nl_N_100000 | Nested logit | 100000 | 4 | None | None | 0.547 | 1.256 | NA | 37.292 | 29.578 | NA | NA | NA | Yes |
| nl_C_3 | Nested logit | 20000 | 3 | None | None | 0.103 | 0.143 | NA | 9.261 | 2.678 | NA | NA | NA | Yes |
| nl_C_10 | Nested logit | 20000 | 10 | None | None | 0.288 | 0.577 | NA | Timeout | 14.117 | NA | NA | NA | Yes |
| nl_C_20 | Nested logit | 20000 | 20 | None | None | 0.827 | 3.791 | NA | Timeout | 62.563 | NA | NA | NA | Yes |
| nl_K_4 | Nested logit | 20000 | 5 | None | None | 0.127 | 0.271 | NA | 53.393 | 4.020 | NA | NA | NA | Yes |
| nl_K_12 | Nested logit | 20000 | 5 | None | None | 0.172 | 0.368 | NA | 49.493 | 9.711 | NA | NA | NA | Yes |
| nl_K_32 | Nested logit | 20000 | 5 | None | None | 0.278 | 0.628 | NA | 194.646 | 46.946 | NA | NA | NA | Yes |
| nl_rho_0p0 | Nested logit | 20000 | 5 | None | None | 0.179 | 0.382 | NA | 56.258 | 9.627 | NA | NA | NA | Yes |
| nl_rho_0p5 | Nested logit | 20000 | 5 | None | None | 0.168 | 0.360 | NA | 48.845 | 11.312 | NA | NA | NA | Yes |
| nl_rho_0p98 | Nested logit | 20000 | 5 | None | None | 0.168 | 0.362 | NA | 47.524 | 10.094 | NA | NA | NA | Yes |
| nl_stress_small | Nested logit | 30000 | 12 | None | None | 0.552 | 1.191 | NA | Timeout | 37.064 | NA | NA | NA | Yes |
| nl_stress_medium | Nested logit | 40000 | 16 | None | None | 1.338 | 6.102 | NA | Timeout | 99.092 | NA | NA | NA | Yes |
| stress_nl_NJK | Nested logit | 50000 | 20 | None | None | 2.781 | 13.058 | NA | Timeout | 227.469 | NA | NA | NA | Yes |

## Objective Diagnostics

- `nl_N_1000`: reference loglike=-5.68e+02; torch_choice ll_diff=0.00e+00, biogeme ll_diff=-3.87e-07, apollo ll_diff=-5.16e-08
- `nl_N_10000`: reference loglike=-5.27e+03; torch_choice ll_diff=0.00e+00, biogeme ll_diff=-3.52e-06, apollo ll_diff=-9.82e-11
- `nl_N_100000`: reference loglike=-5.36e+04; torch_choice ll_diff=5.82e-11, biogeme ll_diff=-1.41e-05, apollo ll_diff=4.07e-10
- `nl_C_3`: reference loglike=-8.29e+03; torch_choice ll_diff=0.00e+00, biogeme ll_diff=-1.13e-05, apollo ll_diff=-9.46e-11
- `nl_C_10`: reference loglike=-1.70e+04; torch_choice ll_diff=0.00e+00, biogeme ll_diff=NA, apollo ll_diff=-4.80e-10
- `nl_C_20`: reference loglike=-2.20e+04; torch_choice ll_diff=-1.09e-11, biogeme ll_diff=NA, apollo ll_diff=-2.91e-11
- `nl_K_4`: reference loglike=-1.62e+04; torch_choice ll_diff=-1.09e-11, biogeme ll_diff=-3.17e-05, apollo ll_diff=2.06e-10
- `nl_K_12`: reference loglike=-7.14e+03; torch_choice ll_diff=-4.55e-12, biogeme ll_diff=-6.80e-05, apollo ll_diff=0.00e+00
- `nl_K_32`: reference loglike=-2.80e+03; torch_choice ll_diff=-4.55e-13, biogeme ll_diff=-1.80e-04, apollo ll_diff=-1.52e-09
- `nl_rho_0p0`: reference loglike=-1.33e+04; torch_choice ll_diff=-3.64e-12, biogeme ll_diff=-2.51e-05, apollo ll_diff=1.33e-10
- `nl_rho_0p5`: reference loglike=-5.90e+03; torch_choice ll_diff=-5.46e-12, biogeme ll_diff=-6.77e-05, apollo ll_diff=-3.46e-11
- `nl_rho_0p98`: reference loglike=-4.45e+03; torch_choice ll_diff=-1.82e-12, biogeme ll_diff=-1.74e-04, apollo ll_diff=-2.93e-09
- `nl_stress_small`: reference loglike=-1.83e+04; torch_choice ll_diff=0.00e+00, biogeme ll_diff=NA, apollo ll_diff=2.44e-10
- `nl_stress_medium`: reference loglike=-2.18e+04; torch_choice ll_diff=-2.91e-11, biogeme ll_diff=NA, apollo ll_diff=0.00e+00
- `stress_nl_NJK`: reference loglike=-2.38e+04; torch_choice ll_diff=5.82e-11, biogeme ll_diff=NA, apollo ll_diff=6.04e-10
