{% set method_name = name.split('.')[-1] %}

FrameOn.preproc.{{ method_name }}
{{ '=' * ('FrameOn.preproc.' + method_name)|length }}

.. currentmodule:: frameon.dataframe.preprocessing

.. automethod:: FrameOnPreproc.{{ method_name }}
