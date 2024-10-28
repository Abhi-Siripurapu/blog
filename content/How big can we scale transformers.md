September 13, 2023
# The Limits of Scaling


I've seen a lot of people recently just post a random graph of FLOPS vs future iterations of GPT-n, assuming we somehow get exponential gains forever. That's pretty clearly not the case, as you'll see through the following napkin math. That obviously begs the question "How big can we get?", which is what I'll try to answer as well.

### Factors at Play
Here's what will factor into this exploration of scaling limits:

- **Energy:** Limited by solar radiation on Earth and heat dissipation capacity—critical constraints for compute power.
- **Compute:** Bound by thermodynamics and intrinsically tied to intelligence—this sets a cap on intelligence capacity.
- **Models:** Serve to estimate intelligence output based on compute input.

### Energy Constraints
Let’s analyze scenarios across increasing energy scales:

1. **$E_0 = 40 \, \text{GW}$** $(4 \times 10^{10} \, \text{W})$: This is about the current energy capacity of U.S. data centers.
2. **$E_1 = 10 \, \text{TW}$** $(1 \times 10^{13} \, \text{W})$: Approximately matches global energy production today.
3. **$E_2 = 1 \, \text{PW}$** $(1 \times 10^{15} \, \text{W})$: Achievable by covering substantial landmass with solar panels.

Nuclear is tempting but faces heat dissipation challenges. Solar, on the other hand, doesn't add heat dissipation burdens to Earth’s ecosystem, making it a more sustainable near-term source. Exceeding $E_2$ would prompt catastrophic warming, so we’ll use it as a sensible upper limit.

#### Calculating $E_2$
Assumptions:
- 20% solar panel efficiency, with 20% land coverage possible.
- Earth's land area is approximately $5 \times 10^{14} \, m^2$, with 168 W/m² energy absorbance and 29% of Earth as land.

$$
E_2 \approx (5 \times 10^{14} \, m^2)(0.29 \times 0.20 \times 0.20)(168 \, W/m^2) \approx 10^{15} \, \text{W}
$$

Increasing solar panel coverage reduces Earth’s reflectivity, with a minor temperature rise of 1-2K.

### Compute Limitations
Compute constraints are abundant (see Wikipedia's "Limits of Computation"). Given our Earth-bound setting, Landauer's Principle applies, since Earth’s temperature is limited by solar irradiance balance.

#### Landauer's Bound for Compute
Landauer’s Principle establishes the minimum energy for erasing a bit:

$$
E_{min} = k_B T \ln 2
$$

At 300K, this translates to:

$$
E_{min} = (1.380649 \times 10^{-23} \, \text{J/K}) \times 300 \, \text{K} \times \ln 2 \approx 2.8707 \times 10^{-21} \, \text{J}
$$

This energy must be dissipated per bit operation. Assuming equilibrium (conservation of energy), the rate of operations per watt becomes:

$$
N_{ops/sec \, per \, watt} = \frac{1}{E_{min}} \approx 3.483 \times 10^{20} \, \text{ops/s/W}
$$

For floating-point operations, with about 50 bit ops per FLOP, we get an upper bound:

$$
5 \times 10^{18} \, \text{FLOPS/W}
$$

This efficiency would be like powering a current top-tier data center on a single watt.

#### Scenarios
1. **$\eta_0 = 10^{13} \, \text{FLOPS/W}$**: Best available compute efficiency today.
2. **$\eta_1 = 5 \times 10^{16} \, \text{FLOPS/W}$**: Achievable at 1% of Landauer’s limit.
3. **$\eta_2 = 5 \times 10^{18} \, \text{FLOPS/W}$**: Landauer’s theoretical maximum.

These cover realistic to boundary scenarios.

### Models and Training Limits
For this analysis, let’s consider the current paradigm for LLM training—expensive and lengthy training with low-cost inference. GPT-4’s training compute estimate sits at $10^{25}$ FLOP, making $10^{27}$ FLOP a plausible next step.

Given an $E_0, \eta_0$ scenario with a 20% compute efficiency for six months:

$$
\approx 1.3 \times 10^{30} \, \text{FLOP}
$$

This aligns with projections for 2030 models, like GPT-6.

#### The "Runaway Improvement" Scenario ($E_1, \eta_1$):
Assuming intelligence-limited tasks are solvable:

$$
\approx 1.6 \times 10^{36} \, \text{FLOP}
$$

This would equate to something around GPT-9’s intelligence, potentially equaling millions of human scientists in capability.

#### World-Covering Scenario ($E_2, \eta_2$):
With massive global reallocation of resources:

$$
\approx 1.6 \times 10^{40} \, \text{FLOP}
$$

GPT-11 training would approach this level.

### Nuclear Energy and Earth’s Heat Dissipation
Using the Stefan-Boltzmann Law, $P = C T^4$, nuclear power scaling can increase Earth’s temperature beyond solar’s contribution. A 50x nuclear output increases Earth to 325K, risking severe climatic and ecological impacts.

### Alternative Tech: Quantum and Reversible Computing
Quantum and reversible computing might eventually break through these limits. Quantum tasks excel narrowly, with uncertain general scaling advantages. Reversible computing could bypass Landauer’s limits but demands wholly new algorithms and hardware.

#### Revisiting the Margolus–Levitin Bound
This quantum limit offers $10^{33}$ FLOPS/W, theoretically achievable with a planetary-scale dilution refrigerator.

### Conclusion
GPT-11 training (at $10^{40} \, \text{FLOP}$) represents a scaling ceiling for terrestrial compute, bound by solar energy, thermodynamics, and the Earth's heat dissipation. Without massive scientific leaps, GPT-9’s level ($10^{36} \, \text{FLOP}$) is more attainable without destabilizing our planet. The final word on scaling? Time to start building a Dyson Sphere.
