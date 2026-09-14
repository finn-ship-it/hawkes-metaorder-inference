# Third-party notices

This repository contains code derived from, or designed to interoperate with,
the following upstream project:

- Project: `konqr/lobSimulations`, maintained by Konark Jain and contributors
- Source: https://github.com/konqr/lobSimulations
- Relevant component: `HawkesRLTrading`
- Audited compatibility revision:
  `f0d5b22a69d9cd0b7d9b3e881514c571c7189e39`

## Use in this repository

- `src/simulator/d2_konark_backend_legacy.py` and
  `src/simulator/d2_konark_latent_book.py` contain an adapted version of
  `HawkesArrival.thinningOgataIS2` from
  `HawkesRLTrading/src/Stochastic_Processes/Arrival_Models.py`. The local
  version applies two documented runtime fixes and otherwise follows the
  upstream method closely.
- `src/simulator/hawkes_core.py` is an independent four-type implementation
  whose thinning-loop structure references the upstream arrival model.
- `src/simulator/layer1_konark_cls.py`,
  `src/simulator/layer1_nonparametric.py`, and `src/simulator/kernels.py`
  document the upstream algorithms or formulae used as references; their
  implementations are adapted to the FX model used in this dissertation.

The complete upstream repository is not included here. Konark Jain gave
permission for the research reuse described above. The upstream project also
requests citation of:

1. Jain, Konark; Firoozye, Nick; Kochems, Jonathan; and Treleaven, Philip.
   “Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes
   Process” (2023). https://doi.org/10.2139/ssrn.4766449
2. Jain, Konark; Firoozye, Nick; Kochems, Jonathan; and Treleaven, Philip.
   “Limit Order Book Simulations: A Review” (2023).
   https://doi.org/10.2139/ssrn.4745587

The notices below are reproduced from the upstream repository and its
`HawkesRLTrading` component. They apply to the corresponding third-party
material only; they do not license the remainder of this repository.

## `lobSimulations` root licence notice

MIT License

Copyright (c) 2023 Daniel Cunha Oliveira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## `HawkesRLTrading` licence notice

MIT License

Copyright (c) [2024] [Alex Kwang]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
