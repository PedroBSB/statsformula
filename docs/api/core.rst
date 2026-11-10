Formulas, terms and design matrices
===================================

.. currentmodule:: statsformula

Formulas
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   formula
   as_formula
   Formula
   as_Formula
   RFormula
   update
   reformulate
   deparse
   parse
   all_vars
   all_names

Terms and model frames
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   terms
   Terms
   drop_terms
   delete_response
   model_frame
   ModelFrame
   model_response
   model_offset
   model_weights
   model_extract
   model_part
   get_all_vars

Design matrices
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   model_matrix
   sparse_model_matrix
   dmatrices
   dmatrix
   ModelMatrix
   SparseModelMatrix
   ModelSpec

Contrasts, bases and options
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   contr_treatment
   contr_sum
   contr_helmert
   contr_poly
   contr_SAS
   ContrastMatrix
   spline_design
   options
   register_function
   set_collation
   get_collation

lavaan syntax
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   lavaan_parse
   lavaanify
   lavaan_partable
   lavaan_names

Datasets
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_data

Errors
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   RError
   FormulaSyntaxError
   LavaanSyntaxError
