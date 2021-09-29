import logging
import shelve


def shelve_full_workspace(filename, var_keys, var_dict: dict):
    """
    Made for debugging

    Should call like:
    shelve_full_workspace(
        filename='shelve.out',
        var_keys=list(dir()),
        var_dict=locals()
    )

    Alternate: var_dict=globals()

    Parameters
    ----------
    filename
    var_keys
    var_dict

    Returns
    -------
    Saves shelf object to disk

    """
    my_shelf = shelve.open(filename, 'n')

    for key in var_keys:
        try:
            my_shelf[key] = var_dict[key]
        except TypeError:
            #
            # __builtins__, my_shelf, and imported modules can not be shelved.
            #
            logging.info(f'ERROR shelving: {key}')
    my_shelf.close()