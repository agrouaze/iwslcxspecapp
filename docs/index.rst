################################################################################
iwslcxspecapp: web application to visualize Sentinel-1 level-1B SLC SAR products
################################################################################

**iwslcxspecapp** is a library to visualize the cross spectrum stored in level 1-B SAR SLC products.

Objets manipulated are `xarray`_.

Sentinel-1 acquisition modes handled by the library: L1B IW

The products are *netCDF* files containing `datatree`_ object.


.. jupyter-execute:: examples/intro.py



Documentation
-------------

Overview
........

    **iwslcxspecapp**  helps to display L1B products (especially IW TOPS Scan SAR acquisitions) containing both intra burst and
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

.. _on github: https://github.com/agrouaze/iwslcxspecapp
.. _datatree: https://github.com/xarray-contrib/datatree