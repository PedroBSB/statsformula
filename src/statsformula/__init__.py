"""statsformula: R model formulas for Python.

A faithful re-implementation of R's formula machinery (``terms``,
``model.frame``, ``model.matrix``, ``update``, ...), the *Formula* package
(multi-part formulas) and the ``mgcv`` smooth terms (``s``, ``te``, ``ti``,
``t2``), validated against R.

>>> import pandas as pd, statsformula as sf
>>> df = pd.DataFrame({"y": [1., 2, 3, 4], "x": [1., 2, 3, 5], "g": list("abab")})
>>> X = sf.model_matrix("y ~ g * log(x)", df)
>>> list(X.columns)
['(Intercept)', 'gb', 'log(x)', 'gb:log(x)']
"""

from ._version import __version__
from ._rtypes import RError, set_collation, get_collation, Factor, Logical, Character, RMatrix
from ._parser import FormulaSyntaxError, parse
from ._terms import Terms
from ._bases import (ContrastMatrix, contr_helmert, contr_poly, contr_SAS, contr_sum,
                     contr_treatment, spline_design)
from ._eval import options, register_function
from ._formula import Formula, RFormula
from ._model import ModelFrame
from ._api import (ModelMatrix, ModelSpec, SparseModelMatrix, all_names, all_vars, as_formula,
                   as_Formula, delete_response, deparse, dmatrices, dmatrix, drop_terms, formula,
                   get_all_vars, model_extract, model_frame, model_matrix, model_offset, model_part,
                   model_response, model_weights, reformulate, sparse_model_matrix, terms, update)
from . import _mgcv  # registers s(), te(), ti(), t2()
from ._mgcv import interpret_gam, smooth_construct, smooth_con, predict_mat, gam_model_matrix, GamSpec, GamPenalty, penalty_matrix
from ._gamfit import gam, GamFit, fit_penalized
from ._lavaan import (LavaanSyntaxError, LavaanWarning, lavaan_names, lavaan_parse, lavaan_partable,
                      lavaanify)

__all__ = [
    "gam", "GamFit", "fit_penalized", "GamPenalty", "penalty_matrix",
    "__version__", "RError", "FormulaSyntaxError", "set_collation", "get_collation",
    "Factor", "Logical", "Character", "RMatrix", "parse", "Terms", "ContrastMatrix",
    "contr_helmert", "contr_poly", "contr_SAS", "contr_sum", "contr_treatment", "spline_design",
    "options", "register_function", "Formula", "RFormula", "ModelFrame", "ModelMatrix", "ModelSpec",
    "SparseModelMatrix", "all_names", "all_vars", "as_formula", "as_Formula", "delete_response",
    "deparse", "dmatrices", "dmatrix", "drop_terms", "formula", "get_all_vars", "model_extract",
    "model_frame", "model_matrix", "model_offset", "model_part", "model_response", "model_weights",
    "reformulate", "sparse_model_matrix", "terms", "update", "interpret_gam", "smooth_construct",
    "GamSpec", "smooth_con", "predict_mat", "gam_model_matrix",
    "lavaan_parse", "lavaanify", "lavaan_partable", "lavaan_names", "LavaanSyntaxError",
    "LavaanWarning",
]
