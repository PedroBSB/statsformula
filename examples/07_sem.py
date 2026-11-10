"""Confirmatory factor analysis through the semopy bridge: Holzinger & Swineford (1939).

The data ship with semopy (``pip install statsformula[sem]``).  The lavaan model is expanded by
``lavaanify()`` into the same parameter table lavaan builds, fitted by semopy, and printed in
lavaan's layout.
"""
from semopy.examples import holzinger39

from statsformula.sem import fit_semopy

hs = holzinger39.get_data()
model = """
  visual  =~ x1 + x2 + x3
  textual =~ x4 + x5 + x6
  speed   =~ x7 + x8 + x9
"""
fit = fit_semopy(model, hs)
print(fit.summary())
