{% set method_name = name.split('.')[-1] %}

FrameOn.explore.{{ method_name }}
{{ '=' * ('FrameOn.explore.' + method_name)|length }}

.. currentmodule:: frameon.dataframe.explore

.. automethod:: FrameOnExplore.{{ method_name }}
