# Plans

Decide Name (Cool Name makes Cool Work)
- LODEN: Lyapunov ODE Networks
- LRNN
- DeLyCe: Deep Lyapunov Certificates
- tbc

Exploit NN capabilities to synthesise Lyapunov functions. 

### Features

- use Neural-ODE, Lagrangian Deep Networks (paper di Nizza con reti RNN che simulano sistemi dinamici) instead of feed-forward nets
- Numeric Verifier, use of Z3 just to confirm the result
- Extra Cegis Components: Simplifier btwn Lrn and Ver and Generaliser btwn Ver and Lrn
  - Simplifier: simplifies expression of V
  - Generaliser: either develops trajectories from the ctx, or gives a parametric ctx
- Automatic Activation Selection (Darboux or Handelmann - inspired)
- Template as counterexamples (parametric ctx?)
- Trajectories as counterexamples

### Difference from LNN

- new loss function: need a continuous loss
- bias = True and last layer w/ weights, not just ones
- support for several activation functions: quadratic, parabolic ramp, linear/quadratic
- support non-poly activations: log, trigonometry 
- support for dReal verifier in combination with non-poly activations

### Meeting Alec/Ale (A Team)

- study SyGUs atchitecture (Elizabeth's work)
- parametric/trajectories counterexamples: how to exploit?
- initialise NN w/ TACAS-like procedure: linearise, synthesise, glue together
- bistability and bistable systems
