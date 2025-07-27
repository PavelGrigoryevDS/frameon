{% set method_name = name.split('.')[-1] %}

SeriesOn.explore.{{ method_name }}
{{ '=' * ('SeriesOn.explore.' + method_name)|length }}

.. currentmodule:: frameon.series.explore

.. automethod:: SeriesOnExplore.{{ method_name }}
