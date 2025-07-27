{% set method_name = name.split('.')[-1] %}

FrameOn.viz.{{ method_name }}
{{ '=' * ('FrameOn.viz.' + method_name)|length }}

.. currentmodule:: frameon.dataframe.visualization

.. automethod:: FrameOnViz.{{ method_name }}
