"""All tunable assumptions in one place.

Every number here is a screening-level assumption, not an engineering value.
Change them for your context; the dashboard reports which values were used.
"""
from dataclasses import dataclass, field, asdict

# UTM zone 44N covers Chennai; all distances and areas are computed in metres.
PROJECTED_CRS = "EPSG:32644"
WGS84 = "EPSG:4326"

# Chennai city-level rainfall cell (IMD 0.25 deg grid).
CHENNAI_LATLON = (13.08, 80.27)


@dataclass
class Neighbourhood:
    name: str
    lat: float
    lon: float
    radius_m: float = 250.0
    held_out: bool = False


NEIGHBOURHOODS = [
    Neighbourhood("mylapore", 13.0339, 80.2697),
    Neighbourhood("t_nagar", 13.0418, 80.2341),
    Neighbourhood("velachery", 12.9791, 80.2209),
    Neighbourhood("anna_nagar", 13.0850, 80.2101, held_out=True),
]


@dataclass
class Hydrology:
    default_runoff_coeff: float = 0.80
    runoff_by_roof: dict = field(default_factory=lambda: {
        "concrete": 0.85, "metal": 0.90, "tile": 0.75,
        "asbestos": 0.80, "thatch": 0.50,
    })
    first_flush_mm: float = 1.0          # lost from every rain day before capture
    floors_default: float = 2.0          # no height data in footprints
    m2_per_person: float = 20.0          # floor area per resident
    demand_lpcd: float = 45.0            # non-potable use (flushing, washing), L/person/day


@dataclass
class Uncertainty:
    # Roof area = footprint area * lognormal(bias, sigma); sigma grows as confidence falls.
    area_bias: float = 0.0
    area_sigma_min: float = 0.05
    area_sigma_slope: float = 0.25       # sigma = min + slope * (1 - confidence)
    # Probability a detected footprint is a real building.
    exist_offset: float = 0.15           # p_exist = clip(confidence + offset, 0, 1)
    osm_confidence: float = 0.90         # OSM footprints carry no score
    n_area_draws: int = 3                # roof draws per rainfall year in optimisation


@dataclass
class Costs:
    # Indian rupees. Rough screening figures, edit for local quotes.
    tank_fixed: float = 25_000.0         # excavation, lid, inlet/outlet, filter
    tank_per_m3: float = 9_000.0         # RCC underground sump
    tank_max_m3: float = 200.0
    hub_pump: float = 30_000.0           # shared tank needs a pump to deliver back
    individual_sizes_m3: tuple = (0.0, 2.0, 5.0, 10.0, 20.0)
    # Plot space limits private sumps: max m3 per m2 of footprint (100 m2 -> 6 m3).
    private_m3_per_roof_m2: float = 0.06
    trench_per_m: float = 400.0          # excavation + road reinstatement
    pipe_per_m: float = 150.0            # 50-63 mm PVC, laid, per line
    lines_per_link: int = 2              # collection line in, supply line back, same trench
    lift_pump: float = 20_000.0          # when roof-to-tank gravity flow fails
    roof_head_m: float = 3.0             # roof outlet height above ground
    pipe_slope: float = 0.01             # 1:100 minimum fall for gravity flow
    max_link_m: float = 250.0            # beyond this, no shared link considered
    discount_rate: float = 0.08
    lifetime_years: int = 20
    om_fraction: float = 0.02            # annual O&M as fraction of capex
    water_value_per_m3: float = 120.0    # avoided tanker water, Rs/m3

    @property
    def crf(self) -> float:
        r, n = self.discount_rate, self.lifetime_years
        return r * (1 + r) ** n / ((1 + r) ** n - 1)

    @property
    def annualise(self) -> float:
        return self.crf + self.om_fraction


@dataclass
class Optimisation:
    periods: int = 365                   # daily water balance in the MILP (73 = 5-day, faster)
    n_scenarios: int = 12                # (year, roof draw) pairs kept in the MILP
    risk_lambda: float = 0.3             # weight on lower-tail CVaR of delivered water
    cvar_alpha: float = 0.25             # worst 25% of scenarios
    hub_fraction: float = 0.2            # candidate hubs as fraction of buildings
    time_limit_s: float = 300.0
    mip_gap: float = 0.01
    failure_threshold: float = 0.8       # cluster fails a year if it delivers < 80% of promise


@dataclass
class Config:
    hydrology: Hydrology = field(default_factory=Hydrology)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    costs: Costs = field(default_factory=Costs)
    opt: Optimisation = field(default_factory=Optimisation)
    scenario_source: str = "generator"   # "generator" (neural rainfall model) or "historical"
    train_years: tuple = tuple(range(1991, 2016))
    test_years: tuple = tuple(range(2016, 2024))
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)
