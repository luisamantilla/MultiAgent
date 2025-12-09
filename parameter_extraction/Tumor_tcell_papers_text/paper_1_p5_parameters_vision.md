# Vision Extraction

- **Model:** gpt-4o
- **Page:** 5
- **Mode:** parameters
- **Note:** Mode: parameters

---

**Bullet List of Parameters:**

1. Infection rate (\(\hat{\imath}\))
2. Uninfected death rate (\(\hat{D}\))
3. Infected death rate (\(\hat{D}\))
4. Recovery rate (\(\hat{R}\))
5. Local immune type (Macrophage, NK cell, CD8\(^+\) T cell)
6. Volume constraint (\(v_t\))
7. Volume multiplier (\(\lambda_x\))
8. Diffusion coefficients (Extracellular virus, Chemokines, Type I IFN, Interleukin 10)
9. Chemotaxis parameters (Macrophage, NK - chemokines, CD8\(^+\) T - chemokines)
10. Adhesion parameters (Uninfected - immune, Infected - immune, Dead - immune, Homotypic immune, Heterotypic immune)

**Detailed Information:**

1. **Infection rate (\(\hat{\imath}\))**
   - **Value:** \(\frac{\beta}{v_t}\)
   - **Units:** Not specified
   - **Context:** Transition rate for infection
   - **Location:** Table 2

2. **Uninfected death rate (\(\hat{D}\))**
   - **Value:** \(\mu_u + a_D\)
   - **Units:** Not specified
   - **Context:** Transition rate for uninfected cell death
   - **Location:** Table 2

3. **Infected death rate (\(\hat{D}\))**
   - **Value:** \(\mu_i + \gamma \left( \frac{s_{i,8,p} R_i H_{A,i}}{v_t} \right) + \gamma \left( \frac{s_{i,8,p} E_i H_{A,i}}{v_t} \right) + \mu_p i\)
   - **Units:** Not specified
   - **Context:** Transition rate for infected cell death
   - **Location:** Table 2

4. **Recovery rate (\(\hat{R}\))**
   - **Value:** \(\alpha_i\)
   - **Units:** Not specified
   - **Context:** Transition rate for recovery
   - **Location:** Table 2

5. **Local immune type**
   - **Macrophage Inflow rate:** \(\eta \left( \frac{s_{m,8,p} R_i H_{A,i}}{v_t} + b_m \mu_m \right)\)
   - **Macrophage Outflow rate:** \(\mu_m\)
   - **NK cell Inflow rate:** \(\eta \left( \frac{s_{n,8,p} R_i H_{A,i}}{v_t} + \mu_k \right)\)
   - **NK cell Outflow rate:** \(\mu_k + b_k \sum \left( \frac{s_{n,8,p}}{v_t} \right) \rho(s)\)
   - **CD8\(^+\) T cell Inflow rate:** \(\frac{\rho_{8,p} v_t}{v_x}\)
   - **CD8\(^+\) T cell Outflow rate:** \(\mu_c + b_c \sum \left( \frac{s_{c,8,p}}{v_t} \right) \rho(s)\)
   - **Units:** Not specified
   - **Context:** Transition rates for local immune response
   - **Location:** Table 2

6. **Volume constraint (\(v_t\))**
   - **Value:** 100 \(\mu m^3\)
   - **Units:** \(\mu m^3\)
   - **Context:** Average cell diameter of 10 \(\mu m\)
   - **Location:** Table 3

7. **Volume multiplier (\(\lambda_x\))**
   - **Value:** 9
   - **Units:** Not specified
   - **Context:** Chosen for an average cell diameter
   - **Location:** Table 3

8. **Diffusion coefficients**
   - **Extracellular virus:** 0.019 \(\mu m^2/s\)
   - **Chemokines:** 1.04 \(\mu m^2/s\)
   - **Type I IFN:** 0.052 \(\mu m^2/s\)
   - **Interleukin 10:** 0.327 \(\mu m^2/s\)
   - **Units:** \(\mu m^2/s\)
   - **Context:** Chosen for diffusion lengths
   - **Location:** Table 3

9. **Chemotaxis parameters**
   - **Macrophage - virus:** 5.000
   - **NK - chemokines:** 5.000
   - **CD8\(^+\) T - chemokines:** 10.000
   - **Units:** Not specified
   - **Context:** Chosen for strong chemotaxis
   - **Location:** Table 3

10. **Adhesion parameters**
    - **Uninfected - immune:** 20
    - **Infected - immune:** 10
    - **Dead - immune:** 10
    - **Homotypic immune:** 25
    - **Heterotypic immune:** 10
    - **Units:** Not specified
    - **Context:** Chosen for preferential attachment
    - **Location:** Table 3