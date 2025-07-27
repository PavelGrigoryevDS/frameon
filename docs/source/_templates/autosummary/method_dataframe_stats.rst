{% set method_name = name.split('.')[-1] %}

FrameOn.stats.{{ method_name }}
{{ '=' * ('FrameOn.stats.' + method_name)|length }}

.. currentmodule:: frameon.dataframe.statistics

.. automethod:: FrameOnStats.{{ method_name }}
