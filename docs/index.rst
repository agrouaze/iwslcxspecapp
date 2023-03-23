############################################################################
l1butils: functions to play with IFREMER L1B Sentinel-1 SLC SAR products
############################################################################

**l1butils** is a library to exploit level 1-B SAR SLC products. Objets manipulated are all `xarray`_.

Acquisition modes available in L1B IFREMER product family are IW and WV.

The products are *netCDF* files containing `datatree`_ object.


.. jupyter-execute:: examples/intro.py



.. image:: oceanspectrumSAR.png
   :width: 500px
   :height: 400px
   :scale: 110 %
   :alt: real part SAR cross spectrum
   :align: right



Documentation
-------------

Overview
........

    **l1butils**  helps to read L1B products (especially IW TOPS Scan SAR acquistions) containing both intra burst and
    inter (i.e. overlapping bursts) burst cross spectrum.


Reference
.........

* :doc:`basic_api`

Get in touch
------------

- Report bugs, suggest features or view the source code `on github`_.

----------------------------------------------

Last documentation build: |today|

.. toctree::
   :maxdepth: 2
   :caption: Home
   :hidden:

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   basic_api

.. _on github: https://github.com/umr-lops/utils_xsarslc_l1b
.. _datatree: https://github.com/xarray-contrib/datatree