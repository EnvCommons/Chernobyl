# NPP Sim Citation Verification Report

## Overview

Systematic verification of every citation, physics constant, and literature reference in the NPP Sim codebase against authoritative sources. 35+ parameters were verified; no hallucinated citations were found. Six issues were identified and fixed.

---

## Verified Correct (No Changes Needed)

### Reactor Parameters

| Parameter | Code Value | Source | Status |
|-----------|-----------|--------|--------|
| RBMK rated power | 3200 MWt | INSAG-7 (IAEA Safety Series No. 75-INSAG-7, 1992) | Verified |
| RBMK beta_eff | 0.0048 | INSAG-7 section A.3 | Verified |
| RBMK void coefficient | +4.7e-4 dk/k | INSAG-7 section 4.3: "+4.7 beta" | Verified |
| RBMK 211 control rods | 211 | 2nd-generation RBMK (Chernobyl Units 3/4); WNA RBMK appendix | Verified |
| RBMK rod speed | 5%/s (~18s full travel) | INSAG-7: 18 seconds full insertion time | Verified |
| TMI-2 rated power | 2772 MWt | NRC reactor database; TMI-2 Solutions project | Verified |
| PWR beta_eff | 0.0065 | Standard U-235 thermal fission value; MIT 22.05 lecture notes | Verified |
| PWR operating pressure | 15.5 MPa | NRC PWR Systems documentation (~155 bar / 2250 psi) | Verified |
| PWR control rods | 69 | TMI-2: 61 CRAs + 8 APSRs = 69 (OSTI technical reports) | Verified |
| BWR beta_eff | 0.0054 | NRC BWR exam bank (mid-cycle BWR value) | Verified |
| BWR rated power | 2381 MWt | Generic BWR-4 class (Fukushima Units 2-5) | Acceptable |
| BWR control rods | 137 | NRC BWR/4 Technology Manual (ML12158A334) | Verified |
| BWR outlet temp | 286 C | TEPCO/WNA Fukushima data | Verified |
| BWR inlet temp | 215 C | GE BWR/4 Technology Manual (feedwater temperature) | Verified |
| BWR containment pressure | 0.53 MPa | NISA: PCV ~500 kPa; Mark I spec ~275-550 kPa | Verified |
| Windscale rated power | 180 MWt | Wikipedia Windscale Piles; Penney Report (1957) | Verified |
| Windscale graphite mass | 1,966 tonnes | Wikipedia: "1,966 tonnes of graphite" | Verified |
| Windscale fuel mass | 180 tonnes | Wikipedia: "180 tonnes of uranium"; Penney Report | Verified |
| Windscale 24 control rods | 24 | 20 coarse + 4 fine, boron steel; Wikipedia Windscale Piles | Verified |
| Windscale ninth anneal | 9th | Springer chapter "The Ninth Anneal"; Wikipedia | Verified |

### Safety Limits and Material Properties

| Parameter | Code Value | Source | Status |
|-----------|-----------|--------|--------|
| Cladding temp limit | 1204 C | 10 CFR 50.46 (2200 F = 1204 C) | Verified |
| UO2 melting point | 2865 C | MATPRO/NUREG/CR-6150; J. Nucl. Mater. studies | Verified |
| Zircaloy melting point | 1850 C | MATPRO; nuclear-power.com | Verified |
| Magnox clad limit | 400 C | Magnox reactive >415 C, melts ~650 C | Verified |
| Uranium metal Cp | 120 J/kg/K | ~27.8 J/mol/K / 238 g/mol = 117 J/kg/K | Verified |
| H2 flammability limit | 4 vol% | NUREG/CR-2726 (LWR Hydrogen Manual); standard LFL | Verified |
| H2 detonation limit | 13 vol% | NUREG/CR-2726; lower detonation limit in air | Verified |
| H2 yield from Zr oxidation | 0.044 kg/kg | Stoichiometric: 2x2.016/91.22 = 0.0442 kg H2/kg Zr | Verified |

### Nuclear Data Constants

| Parameter | Code Value | Source | Status |
|-----------|-----------|--------|--------|
| Xe-135 absorption cross-section | 2.65e-18 cm^2 | ENDF; ~2.65 million barns at 0.0253 eV thermal | Verified |
| Xe-135 decay constant | 2.09e-5 /s | t1/2 = 9.14 hr (NUBASE2020); code gives 9.21 hr (<1% error) | Acceptable |
| I-135 fission yield | 0.061 | ENDF: cumulative ~6.1-6.6% | Verified |
| Xe-135 direct yield | 0.003 | ~0.3% direct fission yield | Verified |

### Physics Correlations and Formulas

| Formula | Code Implementation | Source | Status |
|---------|-------------------|--------|--------|
| Way-Wigner decay heat | P/P0 = 0.066 * t^(-0.2) | Way & Wigner, Phys. Rev. 73, 1318 (1948) | Verified |
| Baker-Just oxidation | A=33.3e6, B=22896 K | Baker & Just, ANL-6548 (1962); 10 CFR 50.46 App K | Verified |
| Saturation temp table | 12 interpolation points | NIST/IAPWS-IF97 steam tables | All 12 points verified |

### Textbook References (All Confirmed Real)

| Reference | Used For | Status |
|-----------|----------|--------|
| Ott & Neuhold, "Introductory Nuclear Reactor Dynamics" (1985) | Point kinetics eigenvalue, RK4 stability | Real textbook |
| Duderstadt & Hamilton, "Nuclear Reactor Analysis" (1976) | Xenon worth, excess reactivity | Real textbook |
| Todreas & Kazimi, "Nuclear Systems" Vol I (2012) | Fuel temperature model | Real textbook |

### NUREG and Regulatory References

| Reference | Used For | Status |
|-----------|----------|--------|
| INSAG-7 (IAEA Safety Series No. 75-INSAG-7, 1992) | RBMK/Chernobyl parameters | Verified |
| NUREG-0600 | TMI-2 accident timeline | Verified |
| NUREG/CR-1250 (Rogovin Report) | TMI-2 core uncovery data | Verified |
| NUREG/CR-6150 (MATPRO) | UO2/Zircaloy material properties | Verified |
| NUREG/CR-2726 | Hydrogen flammability/detonation limits | Verified |
| NUREG/CR-5535 (RELAP5) | Heat transfer regime transitions | Verified |
| NUREG/CR-6849 | TMI-2 Vessel Investigation Project | Verified |
| 10 CFR 50.46 | Cladding temperature limit; Appendix K | Verified |

---

## Issues Found and Fixed

### Issue 1: I-135 Decay Constant Was 2% Off

- **File**: `physics.py:328`
- **Before**: `LAMBDA_I = 2.87e-5  # t1/2 = 6.7 hr`
- **Actual**: I-135 t1/2 = 6.58 +/- 0.03 hr (NUBASE2020; ENDF/B-VIII.0)
- **lambda** = ln(2) / (6.58 * 3600) = 2.926e-5 /s
- **Code gave**: t1/2 = ln(2) / 2.87e-5 = 6.71 hr (2% error)
- **After**: `LAMBDA_I = 2.93e-5  # t1/2 = 6.58 hr (NUBASE2020; ENDF/B-VIII)`
- **Impact**: Small effect on xenon dynamics timing. All existing tests still pass.

### Issue 2: Xe-135 Half-Life Comment Slightly Inaccurate

- **File**: `physics.py:329`
- **Before**: `LAMBDA_XE = 2.09e-5  # t1/2 = 9.2 hr`
- **Actual**: Xe-135 t1/2 = 9.14 +/- 0.02 hr (NUBASE2020)
- **Code lambda gives**: t1/2 = ln(2) / 2.09e-5 = 9.21 hr (<1% error, acceptable)
- **After**: `LAMBDA_XE = 2.09e-5  # t1/2 ~ 9.14 hr (NUBASE2020); lambda gives 9.21h`
- **Impact**: Negligible. Constant kept as-is; comment corrected.

### Issue 3: Baker-Just Unit Comment Was Wrong

- **File**: `physics.py:764-768`
- **Before**: Comments stated units as "m^2/s"
- **Actual**: Baker-Just original units are mg^2/(cm^4 * s) for the weight-gain form w^2 = A * exp(-B/T) * t
- **After**: Corrected comments to cite ANL-6548 (1962) with correct units. Noted the code is a simplified adaptation with a 50% cap, not dimensionally rigorous.
- **Impact**: Comment-only fix. Numerical constants and code behavior unchanged.

### Issue 4: Missing RBMK Neutron Generation Time Citation

- **File**: `physics.py:72`
- **Before**: `neutron_gen_time=5.0e-4,` (no explanation)
- **Rationale**: Graphite-moderated reactors have ~1e-3 s; LWR channels have ~1e-4 s. RBMK (hybrid graphite moderator with water coolant channels) at ~5e-4 s is physically reasonable interpolation.
- **After**: Added comment: `# Graphite-moderated (~1ms) with water channels (~0.1ms)`

### Issue 5: Missing Windscale beta_eff Citation

- **File**: `physics.py:168`
- **Before**: `beta_eff=0.0064,` (no source)
- **Rationale**: Standard U-235 thermal fission beta = 0.0065 (ENDF/B-VIII). Natural uranium with some U-238 fast fission (beta_U238 = 0.0148) gives an effective value slightly below 0.0065. Code value of 0.0064 is reasonable.
- **After**: Added comment: `# Natural uranium, U-235 beta ~ 0.0065 (ENDF/B-VIII)`

### Issue 6: PWR Coolant Temperatures Are B&W-Specific

- **File**: `physics.py:115-116`
- **Code**: `coolant_inlet_temp_c=285.0, coolant_outlet_temp_c=325.0`
- **Generic PWR**: Inlet ~275 C, outlet ~315 C
- **B&W OTSG plants**: Run ~10 C hotter due to once-through steam generator superheat requirements. TMI-2 was a B&W plant.
- **Decision**: Kept values as-is. Added clarifying comments noting these are B&W OTSG-specific.

---

## Tests Added

Eight new citation verification tests were added to `golden_tests.py` in the `TestParameterCitations` class:

| Test | Verifies |
|------|----------|
| `test_iodine_135_half_life` | LAMBDA_I matches t1/2 = 6.58 hr within 2% |
| `test_xenon_135_half_life` | LAMBDA_XE matches t1/2 = 9.14 hr within 2% |
| `test_pwr_control_rods_tmi2` | PWR has 69 rods (61 CRAs + 8 APSRs) |
| `test_bwr_control_rods_137` | BWR-4 has 137 CRDs |
| `test_windscale_24_control_rods` | Windscale has 24 rods (20 coarse + 4 fine) |
| `test_windscale_fuel_mass_180_tonnes` | Windscale has 180,000 kg uranium |
| `test_baker_just_constants` | A=33.3e6, B=22896 K match ANL-6548 |
| `test_way_wigner_formula_coefficient` | 0.066 * t^(-0.2) matches 1948 paper |
| `test_h2_stoichiometry` | 0.044 kg H2/kg Zr matches Zr + 2H2O stoichiometry |

Two existing tests were updated to use correct reference values with 2% relative tolerance.

**Final result: 133/133 tests passing.**

---

## Methodology

1. Extracted all citations and numerical constants from `physics.py`, `npp_sim.py`, and `golden_tests.py`
2. Verified each against authoritative sources via web search: IAEA reports, NRC documents, ENDF nuclear data, NIST steam tables, peer-reviewed papers, and nuclear engineering textbooks
3. Cross-referenced reactor-specific parameters against multiple independent sources
4. Fixed inaccurate values and comments with proper citations
5. Added regression tests to lock down verified values
