{% set method_name = name.split('.')[-1] %}

FrameOn.analysis.{{ method_name }}
{{ '=' * ('FrameOn.analysis.' + method_name)|length }}

.. currentmodule:: frameon.dataframe.analysis

.. automethod:: FrameOnAnalysis.{{ method_name }}
