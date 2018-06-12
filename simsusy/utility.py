

def sin2cos(sin: float)->float:
    """returns Abs[Cos[ArcSin[x]]]"""
    return ((sin + 1) * (-sin + 1)) ** 0.5


cos2sin = sin2cos


def tan2sin(tan: float)->float:
    """returns Abs[Sin[ArcTan[x]]]"""
    return abs(tan) * (tan**2 + 1) ** (-0.5)


def tan2cos(tan: float)->float:
    """returns Abs[Cos[ArcTan[x]]]"""
    return (tan**2 + 1) ** (-0.5)


def sin2tan(sin: float)->float:
    """returns Abs[Tan[ArcSin[x]]]"""
    return abs(sin) * ((-sin + 1) * (sin + 1)) ** (-0.5)


def cos2tan(cos: float)->float:
    """returns Abs[Tan[ArcCos[x]]]"""
    a = cos**(-1)
    return ((a - 1) * (a + 1)) ** 0.5


def tan2costwo(tan: float)->float:
    """returns Abs[Cos[2*ArcTan[x]]]"""
    return abs((tan + 1) * (tan - 1) / (tan**2 + 1))


def tan2sintwo(tan: float)->float:
    """returns Abs[Sin[2*ArcTan[x]]]"""
    return abs(2 * tan / (tan**2 + 1))
