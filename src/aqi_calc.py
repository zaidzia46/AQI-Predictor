PM25_BREAKPOINTS = [  # ug/m3
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [  # ug/m3
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

CO_BREAKPOINTS = [  # ppm
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

SO2_BREAKPOINTS = [  # ppb
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]

NO2_BREAKPOINTS = [  # ppb
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]

O3_BREAKPOINTS = [  # ppm, 8-hour average
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

# Molecular weights (g/mol), used to convert ug/m3 -> ppm/ppb
MOLECULAR_WEIGHTS = {
    "co": 28.01,
    "so2": 64.07,
    "no2": 46.01,
    "o3": 48.00,
}


def ugm3_to_ppm(concentration_ugm3, molecular_weight):
    """Convert ug/m3 to ppm at standard conditions (25C, 1 atm)."""
    if concentration_ugm3 is None:
        return None
    return concentration_ugm3 * 24.45 / (1000 * molecular_weight)


def ugm3_to_ppb(concentration_ugm3, molecular_weight):
    """Convert ug/m3 to ppb at standard conditions (25C, 1 atm)."""
    ppm = ugm3_to_ppm(concentration_ugm3, molecular_weight)
    if ppm is None:
        return None
    return ppm * 1000


def calculate_sub_aqi(concentration, breakpoints):
    """Convert a pollutant concentration into an AQI value (0-500)."""
    if concentration is None:
        return None

    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= concentration <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)

    if concentration > breakpoints[-1][1]:
        return 500
    return None


def calculate_standard_aqi(pm25=None, pm10=None, co=None, so2=None, no2=None, o3=None):
    """
    Calculate the standard AQI from raw pollutant concentrations, all in ug/m3
    (which is what Open-Meteo provides for every pollutant).

    Returns the overall AQI (the worst/highest sub-index), or None if no
    valid pollutant data was available at all.
    """
    sub_indices = []

    sub_indices.append(calculate_sub_aqi(pm25, PM25_BREAKPOINTS))
    sub_indices.append(calculate_sub_aqi(pm10, PM10_BREAKPOINTS))
    sub_indices.append(calculate_sub_aqi(ugm3_to_ppm(co, MOLECULAR_WEIGHTS["co"]), CO_BREAKPOINTS))
    sub_indices.append(calculate_sub_aqi(ugm3_to_ppb(so2, MOLECULAR_WEIGHTS["so2"]), SO2_BREAKPOINTS))
    sub_indices.append(calculate_sub_aqi(ugm3_to_ppb(no2, MOLECULAR_WEIGHTS["no2"]), NO2_BREAKPOINTS))
    sub_indices.append(calculate_sub_aqi(ugm3_to_ppm(o3, MOLECULAR_WEIGHTS["o3"]), O3_BREAKPOINTS))

    valid = [v for v in sub_indices if v is not None]
    if not valid:
        return None
    return max(valid)