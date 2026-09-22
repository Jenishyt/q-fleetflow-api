"""
IMO CII (Carbon Intensity Indicator): rating band A-E for ships >= 5,000 GT.

    CII_attained    = CO2_ttw / (capacity_DWT * distance_nm)
    Required_CII(y) = CII_ref(class) * (1 - 2%/yr through 2026)

Vessels under 5,000 GT don't technically fall under CII - handled via a
"CII-lite" report-only toggle so the engine stays honest about scope
(plan Sec 8.3).
"""
from __future__ import annotations
from dataclasses import dataclass
from src.compliance.fueleu import load_factors


@dataclass
class CIIResult:
    attained: float           # gCO2/(t*nm)
    required: float           # gCO2/(t*nm)
    ratio: float              # attained / required - <=1.0 is compliant
    rating: str                # A-E
    enforced: bool             # False if vessel is under the GT threshold (CII-lite)


def required_cii(vessel_class: str, year: int, factors: dict | None = None) -> float:
    factors = factors or load_factors()
    cii_cfg = factors["cii"]
    ref = cii_cfg["cii_ref_by_class"][vessel_class]
    annual_reduction = cii_cfg["annual_reduction_through_2026"]
    years_from_base = max(0, min(year, 2026) - 2019)  # IMO base year 2019
    reduction = annual_reduction * years_from_base
    return ref * (1 - reduction)


def _rating_band(ratio: float) -> str:
    # simplified banding (real IMO bands use class-specific d1-d4 vectors;
    # this is a linear approximation, documented as such)
    if ratio <= 0.85:
        return "A"
    if ratio <= 0.95:
        return "B"
    if ratio <= 1.05:
        return "C"
    if ratio <= 1.15:
        return "D"
    return "E"


def cii_check(
    co2_ttw_g: float,
    capacity_dwt: float,
    distance_nm: float,
    vessel_class: str,
    year: int,
    vessel_gt: float | None = None,
    factors: dict | None = None,
) -> CIIResult:
    factors = factors or load_factors()
    threshold = factors["cii"]["applies_gt_threshold"]
    enforced = vessel_gt is None or vessel_gt >= threshold

    attained = co2_ttw_g / (capacity_dwt * distance_nm)
    required = required_cii(vessel_class, year, factors)
    ratio = attained / required
    rating = _rating_band(ratio)

    return CIIResult(attained=attained, required=required, ratio=ratio,
                      rating=rating, enforced=enforced)
