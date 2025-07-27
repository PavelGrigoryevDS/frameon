{% set method_name = name.split('.')[-1] %}

SeriesOn.preproc.{{ method_name }}
{{ '=' * ('SeriesOn.preproc.' + method_name)|length }}

.. currentmodule:: frameon.series.preprocessing

.. automethod:: SeriesOnPreproc.{{ method_name }}
