NUMBER_BY_SYMBOL = {'H': 1, 'C': 6, 'N': 7, 'O': 8}

SYMBOL_BY_NUMBER = {value: key for key, value in NUMBER_BY_SYMBOL.items()}


class Standardizer:
    def __init__(self, mean, std, task='regression'):
        if task == 'regression':
            self.mean = mean
            self.std = std
        elif task == 'classification':
            self.mean = 0
            self.std = 1

    def __call__(self, x, rev=False):
        if rev:
            return (x * self.std) + self.mean
        return (x - self.mean) / self.std


class ConverterError(Exception):
    """
    An exception raised when converting between molecule representations.
    """
    pass


def xyz_file_format_to_xyz(xyz_file):
    """
    Creates a xyz dictionary from an `XYZ file format <https://en.wikipedia.org/wiki/XYZ_file_format>`_ representation.

    Args:
        xyz_file (str): The content of an XYZ file

    Returns:
        dict: An xyz dictionary.

    Raises:
        ConverterError: If cannot identify the number of atoms entry or if it is different that the actual number.
    """

    lines = xyz_file.strip().splitlines()
    if not lines[0].isdigit():
        raise ConverterError('Cannot identify the number of atoms from the XYZ file format representation. '
                             'Expected a number, got: {0} of type {1}'.format(lines[0], type(lines[0])))
    number_of_atoms = int(lines[0])
    lines = lines[2:]
    if len(lines) != number_of_atoms:
        raise ConverterError('The actual number of atoms ({0}) does not match the expected number parsed ({1}).'.format(
            len(lines), number_of_atoms))
    xyz_str = '\n'.join(lines)

    xyz_dict = {'symbols': tuple(), 'isotopes': tuple(), 'coords': tuple()}
    for line in xyz_str.strip().splitlines():
        if line.strip():
            splits = line.split()
            if len(splits) != 4:
                raise ConverterError(f'xyz_str has an incorrect format, expected 4 elements in each line, '
                                     f'got "{line}" in:\n{xyz_str}')
            symbol = splits[0]
            if '(iso=' in symbol.lower():
                isotope = int(symbol.split('=')[1].strip(')'))
                symbol = symbol.split('(')[0]
            else:
                # no specific isotope is specified in str_xyz
                isotope = NUMBER_BY_SYMBOL[symbol]
            coord = (float(splits[1]), float(splits[2]), float(splits[3]))
            xyz_dict['symbols'] += (symbol,)
            xyz_dict['isotopes'] += (isotope,)
            xyz_dict['coords'] += (coord,)
    return xyz_dict
