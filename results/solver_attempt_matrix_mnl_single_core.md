# Solver Attempt Matrix

Each benchmark case is attempted with every configured solver where an aligned wrapper exists. Runtimes report estimation plus covariance on one logical CPU. A dagger marks a clearly inferior final log likelihood; its runtime is retained but its estimate is excluded from consistency.
`ok` means the solver completed the aligned case, `failed` means the wrapper was attempted but failed, and `unsupported` means no aligned wrapper exists for that solver/model case.

| case | dataset | model | torchdcm | torch-choice | scipy_bfgs | biogeme | apollo | mlogit | gmnl | xlogit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| swissmetro_mnl | Swissmetro | MNL | ok (0.010s) | ok (0.014s) | ok (2.366s) | ok (1.346s) | ok (0.725s) | ok (0.123s) | failed | ok (0.026s) |
| nhts_2022_mnl | NHTS 2022 | MNL | ok (0.095s) | ok (0.385s) | ok (38.723s) | ok (2.749s) | ok (7.617s) | ok (1.388s) | ok (18.450s) | ok (0.713s) |
| biogeme_public_airline | airline | MNL | ok (0.007s) | ok (0.010s) | ok (3.150s) | ok (1.573s) | ok (0.330s) | ok (0.052s) | ok (0.107s) | ok (0.011s) |
| biogeme_public_parking | parking | MNL | ok (0.007s) | ok (0.009s) | ok (1.930s) | ok (1.503s) | ok (0.286s) | ok (0.031s) | ok (0.056s) | ok (0.006s) |
| biogeme_public_telephone | telephone | MNL | ok (0.005s) | ok (0.006s) | ok (0.242s) | ok (1.652s) | ok (0.253s) | ok (0.021s) | failed | ok (0.004s) |
| biogeme_public_lpmc | lpmc | MNL | ok (0.088s) | ok (0.201s) | ok (56.996s) | ok (1.575s) | ok (12.044s) | ok (2.020s) | ok (5.644s) | ok (0.334s) |
| mlogit_catsup | catsup | MNL | ok (0.004s) | ok (0.009s) | ok (0.376s) | ok (1.592s) | ok (0.271s) | ok (0.049s) | ok (0.049s) | ok (0.008s) |
| mlogit_cracker | cracker | MNL | ok (0.009s) | ok (0.017s) | ok (0.545s) | ok (1.570s) | ok (0.275s) | ok (0.055s) | ok (0.056s) | ok (0.009s) |
| mlogit_electricity | electricity | MNL | ok (0.012s) | ok (0.021s) | ok (2.861s) | ok (2.031s) | ok (0.406s) | ok (0.070s) | ok (0.242s) | ok (0.013s) |
| mlogit_hc | hc | MNL | ok (0.004s) | ok (0.005s) | ok (0.048s) | ok (2.157s) | ok (0.215s) | ok (0.014s) | ok (0.017s) | ok (0.002s) |
| mlogit_heating | heating | MNL | ok (0.003s) | ok (0.004s) | ok (0.141s) | ok (1.674s) | ok (0.220s) | ok (0.023s) | ok (0.022s) | ok (0.003s) |
| mlogit_mode | mode | MNL | ok (0.003s) | ok (0.003s) | ok (0.201s) | ok (1.615s) | ok (0.209s) | ok (0.013s) | ok (0.015s) | ok (0.002s) |
| mlogit_nox | nox | MNL | ok (0.005s) | ok (0.008s) | ok (0.236s) | ok (5.146s) | ok (0.286s) | ok (0.035s) | failed | ok (0.004s) |
| mlogit_risky_transport | risky_transport | MNL | ok (0.025s) | ok (0.029s) | ok (1.139s) | ok (2.325s) | ok (0.346s) | ok (0.040s) | failed | failed |
| mlogit_train | train | MNL | ok (0.016s) | ok (0.019s) | ok (1.766s) | ok (1.359s) | ok (0.286s) | ok (0.031s) | ok (0.047s) | ok (0.005s) |
| mlogit_fishing | fishing | MNL | ok (0.011s) | ok (0.015s) | ok (0.815s) | ok (1.543s) | ok (0.274s) | ok (0.941s) | ok (0.549s) | ok (0.006s) |
| mlogit_modecanada | modecanada | MNL | ok (0.009s) | ok (0.014s) | ok (1.967s) | ok (1.463s) | ok (0.332s) | ok (0.521s) | failed | failed |

## Failures

- `swissmetro_mnl` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
- `biogeme_public_telephone` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
- `mlogit_nox` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
- `mlogit_risky_transport` / `gmnl`: Error in s + x[[i]] : non-conformable arrays
Calls: gmnl ... eval -> eval -> maxNRCompute -> fn -> fnOrig -> suml
Execution halted
- `mlogit_risky_transport` / `xlogit`: ValueError: inconsistent alts values in long format
- `mlogit_modecanada` / `gmnl`: NA NA NA NA NA NA NA NA NA Error in s + x[[i]] : non-conformable arrays
- `mlogit_modecanada` / `xlogit`: NA NA NA NA NA NA NA NA NA xlogit MultinomialLogit requires consistent alternatives in long format; ModeCanada has ragged choice sets.
