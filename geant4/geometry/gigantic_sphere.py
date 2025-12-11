
def get_design(mag_field, sens_film = None, material="G4_Fe"):
    detector = {
        # "worldPositionX": 0, "worldPositionY": 0, "worldPositionZ": 0, "worldSizeX": 11, "worldSizeY": 11,
        # "worldSizeZ": 100,
        # "magnets": magnets,
        "material": material,
        "magnetic_field": mag_field,
        "type": 3,
        "store_all": True,
        "limits": {
            "max_step_length": -1,
            "minimum_kinetic_energy": -1
        }
    }
    if sens_film is not None:
        detector["sensitive_film"] = sens_film
    return detector