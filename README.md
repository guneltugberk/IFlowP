# IFlowP IPR Prediction and Construction

This Python project implements a series of reservoir engineering correlations to compute various gas properties and gas well flow rates. The code is designed to evaluate the gas compressibility factor (z‑factor), formation volume factor (Bg), gas density, viscosity, and ultimately the gas flow rate (Qg) through a reservoir using both Darcy and non‑Darcy (turbulent) flow correlations.

The methods implemented here are based on well‑established correlations in reservoir engineering such as the Dranchuk–Abou-Kassem correlation for z‑factor, the Beggs–Brill correlation as a fallback, and flow equations commonly found in gas well performance analyses.

These results are then used to feed the PINN algorithm for Back Pressure Equation defined by Rawlins et. all [2]. This equation was firstly developed in 1935, and it was based on emprical observations from the gas wells in the United States. In our project, it was aimed to fine tune the coefficient of the Back Pressure equation. 

---

## Table of Contents

- [Project Overview](#project-overview)
- [Installation and Dependencies](#installation-and-dependencies)
- [Usage](#usage)
- [Project Structure and Code Explanation](#project-structure-and-code-explanation)
  - [GasProperties Class](#gasproperties-class)
  - [GasFlow Class](#gasflow-class)
- [Key Equations and Their References](#key-equations-and-their-references)
- [Bibliography](#bibliography)

---

## Project Overview

This project is intended for reservoir engineers and students interested in understanding gas behavior in reservoirs. The code performs the following tasks:

- **Initialization and Validation:** The `GasProperties` class sets up the basic gas properties based on the specific gravity (γ), pressure, and temperature. It validates that the specific gravity lies within acceptable limits.
- **Gas Property Calculations:** Using established correlations, the code calculates:
  - The gas molecular weight and pseudocritical properties.
  - The z‑factor using the Dranchuk–Abou-Kassem correlation (with a fallback to the Beggs–Brill method if conditions are out of range).
  - The gas formation volume factor (Bg) based on pressure, temperature, and z‑factor.
  - Gas density and viscosity using correlations (e.g., Lee’s method).
- **Flow Rate Determination:** The `GasFlow` class extends `GasProperties` and includes methods to:
  - Update the flowing bottom‑hole pressure.
  - Compute the real‑gas pseudo-pressure difference by integrating the function  
    $$\Delta \psi=\int_{P_{wf}}^{P_{res}}\frac{2p}{\mu_g\,z}\,dp $$  
    which is a core part of the gas flow rate calculation [1].
  - Determine the single‑phase gas flow rate (Qg) using both Darcy (linear) and turbulent (non‑Darcy) flow equations. For example, the Darcy flow rate is approximated as:  
    $$ Q_g = \frac{k\,h\,\Delta \psi}{1422\,T\left(\ln\left(\frac{r_e}{r_w}\right)-0.75+s\right)} $$  
    where $$\Delta \psi$$ is the pseudo‑pressure difference [3].

---

## Installation and Dependencies

This project requires Python 3 and the following Python libraries:
- **math** – For basic mathematical operations.
- **scipy** – Specifically, `fsolve` (from `scipy.optimize`) for solving non‑linear equations and `quad` (from `scipy.integrate`) for numerical integration.

To install SciPy (if not already installed), run:

```bash
pip install scipy
```

---

## Usage

1. **Importing the Module:**  
   You can import the module in your Python script:
   ```python
   from gas_flow_module import GasProperties, GasFlow
   ```

2. **Creating an Object:**  
   For example, to calculate gas properties:
   ```python
   # Initialize with specific gravity, reservoir pressure, and temperature.
   gp = GasProperties(gamma=0.65, Pressure=3000, Temperature=150)
   ```

3. **Calculating the z‑factor and Formation Volume Factor:**  
   ```python
   z = gp.z_factor()
   Bg = gp.Bg()
   ```
#### Beggs & Brill Fallback

Uses the formula:

$$
z = A + \frac{(1 - A)}{\exp(B)} + C \, (P_{pr})^D
$$

with constants \(A\), \(B\), \(C\), and \(D\) as coorelation coeficient.  
**Reference:** Also detailed in [1].


4. **Flow Rate Calculation:**  
   To calculate the gas flow rate, create a `GasFlow` object (which requires a flowing bottom‑hole pressure `Pwf`) and then call:
   ```python
   gf = GasFlow(gamma=0.65, Pressure=3000, Temperature=150, Pwf=2500)
   flow_rate = gf.gas_flow_rate(k=50, h=10, re=1000, rw=0.3, skin=0, phi=0.15, turbulant='Yes')
   ```
#### Lee et al. Gas Viscosity

The `viscosity_gas()` function uses:

$$
\mu_g = K \, \exp\!\Bigl(x \,\rho^y \Bigr)
$$

where \rho is the gas density at reservoir conditions, and \(K\), \(x\), \(y\) are correlation parameters.  
**Reference:** [1].

#### Non-Darcy (Turbulent) Flow Coefficients

When `turbulant='Yes'`, a rate-dependent skin term is included. For example:

$$
\beta = \frac{4.85 \times 10^4}{\phi^{5.5} \sqrt{k}} \quad \text{[3]}
$$

$$
D = \left(\frac{2.22 \times 10^{-15} \,\gamma_g}{\mu_g \, r_w \, h}\right)\,\beta\,k \quad \text{[3]} 
$$

**Reference:** These equations are found in [3].

5. **Updating Pressure or Pwf:**  
   Use the `update_pres(new_pr)` and `update_pwf(new_pwf)` methods to update reservoir and bottom‑hole pressures without re‑instantiating the objects.
 
6. **Condition**  
 If Dranchuk & Abou-Kassem equation did not converge the code will be automatically using Beggs & Brill 


---

## Project Structure and Code Explanation

### GasProperties Class

The `GasProperties` class is the core of the module. It includes:

- **Initialization (`__init__`):**  
  Sets the gas specific gravity, calculates molecular weight, and determines pseudocritical properties using correlations (e.g.),  
  $$ p_{pc} = 756.8 - 131\gamma - 3.6\gamma^2 $$  
  and  
  $$ T_{pc} = 169.2 + 349.5\gamma - 74\gamma^2 $$  
   These correlations are common in reservoir engineering texts [1].

- **z‑factor Calculation (`z_factor`):**  
  Uses the Dranchuk–Abou-Kassem correlation. If the reservoir pressure and temperature fall outside the valid range, it falls back on the Beggs–Brill correlation [1].  
  This step is critical since the z‑factor affects all subsequent calculations.

- **Gas Formation Volume Factor (`Bg`):**  
  Calculates the gas formation volume factor with the equation:  
  $$ Bg = 0.005035\,\frac{z\,T_{rankine}}{P} $$  
  which reflects the relationship between pressure, temperature, and gas compressibility [1].

- **Density and Viscosity:**  
  - The `_density_gas_lee` and `gas_density` methods compute gas density using standard correlations.
  - The `viscosity_gas` method calculates gas viscosity through an exponential function of density (Lee’s correlation), which is common in petrophysical evaluations [1].

- **Real-Gas Pseudo‑Pressure (`real_gas_pseudo_pressure`):**  
  This method numerically integrates the function:  
  $$\Delta \psi=\int_{P_{wf}}^{P_{res}}\frac{2p}{\mu_g\,z}\,dp $$  
  to obtain the pseudo‑pressure difference used in flow rate calculations [1].

### GasFlow Class

The `GasFlow` class inherits from `GasProperties` and adds functionality to compute the gas flow rate:

- **Initialization:**  
  In addition to the properties inherited from `GasProperties`, it requires the flowing bottom‑hole pressure (`Pwf`).

- **Flow Rate Calculation (`gas_flow_rate`):**  
  This method computes the gas flow rate \( Q_g \) in MSCF/D. It performs the following steps:
  - **Pseudo‑Pressure Difference:**  
    Determines the difference in real‑gas pseudo‑pressure between the reservoir and the bottom‑hole.
  - **Darcy Flow (Linear Flow):**  
    For laminar (Darcy) flow, the rate is calculated as:  
    $$ Q_g = \frac{k\,h\,\Delta \psi}{1422\,T\left(\ln\left(\frac{r_e}{r_w}\right)-0.75+s\right)} \quad \text{[3]}  $$  
    which follows standard gas well performance equations.
  - **Non‑Darcy (Turbulent) Flow:**  
    When turbulent flow is indicated (by the `turbulant` flag), the code introduces an additional coefficient based on the Beta correlation (see Eq. 7.116b in [3]) and solves a quadratic equation to account for the non‑Darcy behavior.


$$
\beta = \frac{4.85 \times 10^4}{\phi^{5.5} \sqrt{k}} \quad \text{[3]}
$$

$$
D = \left(\frac{2.22 \times 10^{-15} \,\gamma_g}{\mu_g \, r_w \, h}\right)\,\beta\,k \quad \text{[3]} 
$$

$$ 
Q_g = \frac{k\,h\,\Delta \psi}{1422\,T\left(\ln\left(\frac{r_e}{r_w}\right)-0.75+s + DQ_g\right)} \quad \text{[3]} 
$$ 


- **Pressure Update Methods:**  
  The class also includes methods to update the bottom‑hole pressure (`update_pwf`), ensuring flexibility during simulation.

---

## Key Equations and Their References

1. **Real‑Gas Pseudo‑Pressure Integration:**  
   $$ m(p)=\int_{P_{wf}}^{P_{res}}\frac{2p}{\mu_g\,z}\,dp \quad \text{[1]} $$

2. **Gas Formation Volume Factor:**  
   $$ Bg = 0.005035\,\frac{z\,T_{rankine}}{P} \quad \text{[1]} $$

3. **Darcy Flow Rate for Gas Wells:**  
   $$ Q_g = \frac{k\,h\,\Delta \psi}{1422\,T\left(\ln\left(\frac{r_e}{r_w}\right)-0.75+s\right)} \quad \text{[3]} $$

4. **Non-Darcy Flow Rate for Gas Wells:**
  $$ Q_g = \frac{k\,h\,\Delta \psi}{1422\,T\left(\ln\left(\frac{r_e}{r_w}\right)-0.75+s + DQ_g\right)} \quad \text{[3]} $$ 
  
5. **Pseudocritical Properties:**  
   $$ p_{pc} = 756.8 - 131\gamma - 3.6\gamma^2 \quad \text{[1]} $$  
   $$ T_{pc} = 169.2 + 349.5\gamma - 74\gamma^2 \quad \text{[1]} $$

6. **Back Pressure Equation:**
  $$ Q_g = C \bigl(\psi(P_r) - \psi(P_{wf})\bigr)^n \quad \text{[3]} $$

These equations form the backbone of the module’s calculations and are directly referenced from standard gas reservoir engineering texts.

---

## Bibliography

1. Ahmed, T. (2001). *Reservoir Engineering Handbook*. Butterworth-Heinemann.
2. Rawlins, E.L. and Schellhardt, M.A. 1935. Backpressure Data on Natural Gas Wells and Their Application to Production Practices, 7. Monograph Series, U.S. Bureau of Mines.
3. Tiab, D., & Donaldson, E. C. (2004). *Petrophysics: Theory and Practice of Measuring Reservoir Rock and Fluid Transport Properties* (2nd ed.). Elsevier.
---
