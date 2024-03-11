Getting started
===============

.. note::

   This section is under construction.

To do: put content of readme here and link other files.
..  # Getting started
    #### Dependencies
    `biolgca` heavily depends on `numpy` and `matplotlib`. To install all dependencies 
    for using the package, run this from a command line:
    ```python
    pip install matplotlib==3.3.2 numpy scipy sympy
    ```
    (On Windows, a terminal that understands `pip` out of the box can be opened in Anaconda in the 
    "Environments" tab, clicking on the triangle next to the environment's name.)

    If you want to add code to the package, in order to run tests and build documentation you should also install:
    ```python
    pip install pytest==6.2.5 Sphinx==4.4.0 sphinx-autodoc-typehints alabaster numpydoc
    ```

    #### Installation
    `biolgca` does not have a package distribution yet. To use it, clone (or unzip the download of) 
    the master branch of the repository into a folder of your choice.

    If your scripts are not going to be in the `biolgca` folder, add the following to
    the beginning of the Python files:
    ```python
    import sys
    sys.path.insert(1, "/absolute/path/to/folder/biolgca")
    ```

    #### Use
    To use the package, simply import the `get_lgca` function:
    ```python
    from lgca import get_lgca
    ```
    It will return an instance of the correct class of LGCA according to the passed 
    arguments, e.g. a 1D identity-based LGCA (`IBLGCA_1D`), already initialised with 
    the specified initial conditions. This can be used to simulate 
    the automaton with the specified interaction and inspect the results.
    ```python
    # request the LGCA
    lgca = get_lgca(ib=True, geometry='1d', interaction='random_walk')
    # simulate for 50 timesteps
    lgca.timeevo(timesteps=50)
    # plot the development of the particle/cell density over time
    lgca.plot_density()
    ```

    This [Tutorial](./BioLGCA.ipynb) guides you through the argument options.
