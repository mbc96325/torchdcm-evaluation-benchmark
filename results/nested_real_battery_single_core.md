# Real-data Nested Logit Benchmark

Runtimes report estimation plus covariance on one logical CPU.
A dagger marks a clearly inferior final log likelihood; its runtime is retained but its estimate is excluded from consistency. N.A. means fewer than two comparable estimates remain.

| Data | Model | N | TorchDCM | Torch-Choice | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Swissmetro | Nested logit | 10719 | 0.054 | 0.074 | 5.645 | 1.078 | Yes |
| LPMC London | Nested logit | 81086 | 0.662 | 1.519 | 23.963 | 13.544 | Yes |
| NHTS 2022 | Nested logit | 27375 | 0.457 | 1.004 | 104.413 | Fail | Yes |
| Parking Spain | Nested logit | 1576 | 0.026 | 0.032 | 6.555 | 0.391 | Yes |
| Airline itinerary | Nested logit | 3609 | 0.059 | 0.076 | 6.523 | 0.389 | Yes |
| Catsup | Nested logit | 2798 | 0.043 | 0.064 | 10.606 | 0.345 | Yes |
| Cracker | Nested logit | 3292 | 0.031 | 0.046 | 10.573 | 0.378 | Yes |
| Electricity | Nested logit | 4308 | 0.082 | 0.144 | 20.628 | Fail | Yes |
| Fishing | Nested logit | 1182 | 0.058 | 0.084 | 9.708 | 0.324 | Yes |
| HC | Nested logit | 250 | 0.063 | 0.079 | 34.202 | 0.322 | Yes |
| Heating | Nested logit | 900 | 0.060 | 0.080 | 10.125 | 0.306 | Yes |
| Mode | Nested logit | 453 | 0.027 | 0.035 | 9.796 | 0.295 | Yes |

## Nest specifications

- `swissmetro_nested`: PUBLIC=TRAIN,SM; PRIVATE=CAR
- `lpmc_nested`: ACTIVE=walk,cycle; MOTORIZED=pt,drive
- `nhts_2022_nested`: ACTIVE=WALK,BIKE; MOTORIZED=AUTO,TRANSIT,OTHER
- `parking_nested`: FACILITY=FSP,PSP; PUP_NEST=PUP
- `airline_nested`: ALT12=ALT1,ALT2; ALT3_NEST=ALT3
- `mlogit_catsup_nested`: HEINZ=heinz41,heinz32,heinz28; HUNTS=hunts32
- `mlogit_cracker_nested`: NATIONAL=sunshine,kleebler,nabisco; PRIVATE=private
- `mlogit_electricity_nested`: PLAN_12=1,2; PLAN_34=3,4
- `mlogit_fishing_nested`: SHORE=beach,pier; BOAT=boat,charter
- `mlogit_hc_nested`: CURRENT=gcc,ecc,erc; NEW=gc,ec,er; HPC_NEST=hpc
- `mlogit_heating_nested`: GAS=gc,gr; ELECTRIC=ec,er; HEATPUMP=hp
- `mlogit_mode_nested`: PRIVATE=car,carpool; TRANSIT=bus,rail
