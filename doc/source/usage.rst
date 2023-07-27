Usage
=====

.. _installation:

Installation
------------

To use pyNICAM-DC, first install it using pip:

.. code-block:: console

   (.venv) $ pip install pynicamdc

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``pynicamdc.Pynicamdc.setup()`` method:

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`Pynicamdc.setup()`
will raise an exception.


For example:

>>> from pynicamdc import pynicamdc
>>> dc = pynicamdc.Pynicamdc()
>>> dc.setup()
>>> dc.run()
