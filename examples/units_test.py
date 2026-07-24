from classical_diffusion.system import (
    PeriodicSystem1D,
    UnitSystem,
)

system = PeriodicSystem1D(
    gamma=4e12,
    temperature=110,
    m=8e-27,
    delta_x=3e-10,
    barrier_energy=1.6e-22,
    units=UnitSystem(),
)
print(system.units)
# Convert to the system's units
print(system.with_system_units(system.simulation_units()))

# Convert back to regular units
print(system.with_system_units(UnitSystem()))
