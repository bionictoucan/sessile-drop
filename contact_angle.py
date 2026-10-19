import torch, math
from typing import Tuple, Union


def fit_ellipse(
    x: torch.tensor, y: torch.tensor, device: Union[torch.device, str, int]
) -> torch.tensor:
    """
    This function fits an ellipse to a set of (x,y)-coordinates describing the
    edge of the droplet and returns the fitting parameters for the Cartesian
    ellipse equation.

    ..math::
        ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0

    Parameters
    ----------
    x : torch.tensor
        The x-coordinates of the ellipse.
    y : torch.tensor
        The y-coordinates of the ellipse.

    Returns
    -------
     : torch.tensor
        The fitted parameters for the Cartesian ellipse equation.
    """

    D1 = torch.vstack([x**2, x * y, y**2]).T.double()
    D2 = torch.vstack([x, y, torch.ones(x.shape[0], device=device)]).T.double()
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -torch.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = torch.tensor(
        ((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=torch.double, device=device
    )
    M = torch.linalg.inv(C) @ M
    eigval, eigvec = torch.linalg.eig(M)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, torch.nonzero(con.double() > 0)[0]].double()

    return torch.concatenate((ak, T @ ak)).ravel()


def cart_to_pol(coeffs: torch.tensor) -> Tuple[float]:
    """
    Converts the Cartesian parameters of the ellipse (a, 2b, c, 2d, 2f, g) to
    polar parameters (x0, y0, ap, bp, e, phi) where (x0,y0) is the ellipse
    centre, (ap,bp) is the semi-major and semi-minor axes, e is the eccentricity
    of the ellipse and phi is the rotation of the semi-major axis from the
    x-axis.

    Parameters
    ----------
    coeffs : torch.tensor
        The fitted Cartesian coefficients of the ellipse.

    Returns
    -------
    x0 : float
        The x-coordinate of the centre of the fitted ellipse.
    y0 : float
        The y-coordinate of the centre of the fitted ellipse.
    ap : float
        The semi-major axis of the fitted ellipse.
    bp : float
        The semi-minor axis of the fitted ellipse.
    e : float
        The eccentricity of the fitted ellipse.
    phi : float
        The angle between the semi-major axis and the x-axis.
    """
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a * c
    if den > 0:
        raise ValueError(
            "Coeffs do not represent an ellipse: b^2 - 4ac must be negative!"
        )

    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den

    num = 2 * (a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g)
    fac = torch.sqrt((a - c) ** 2 + 4 * b**2)

    ap = torch.sqrt(num / den * (1 / (fac - a - c)))
    bp = torch.sqrt(num / den * (1 / (-fac - a - c)))

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r
    e = torch.sqrt(1 - r)

    if b == 0:
        phi = 0 if a < c else math.pi / 2
    else:
        phi = torch.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += math.pi / 2

    if not width_gt_height:
        phi += math.pi / 2

    phi = phi % math.pi

    return x0, y0, ap, bp, e, phi


def get_ellipse_pts(
    params: torch.tensor, npts: int = 100, tmin: float = 0, tmax: float = 2 * math.pi
) -> Tuple[torch.tensor, torch.tensor]:
    """
    This function generates the points along the ellipse at `npts`
    linearly-spaced points in the parameterised dimension (t).

    Parameters
    ----------
    params : torch.tensor
        The Cartesian parameters of the fitted ellipse.
    npts: int, optional
        The number of points to sample along the ellipse in the parameterised
        dimension direction. Default is 100.
    tmin: int, optional
        The start point of the linearly-spaced points in the parameterised
        dimension. Default is 0.
    tmax: int, optional
        The end point of the linearly-spaced points in the parameterised
        dimension. Default is 2*pi.

    Results
    -------
    x : numpy.ndtorch.tensor or cupy.ndtorch.tensor
        The x-coordinates of the fitted ellipse.
    y : numpy.ndtorch.tensor or cupy.ndtorch.tensor
        The y-coordinates of the fitted ellipse.
    """
    x0, y0, ap, bp, e, phi = cart_to_pol(params)

    t = torch.linspace(tmin, tmax, npts)
    x = x0 + ap * torch.cos(t) * torch.cos(phi) - bp * torch.sin(t) * torch.sin(phi)
    y = y0 + ap * torch.cos(t) * torch.sin(phi) + bp * torch.sin(t) * torch.cos(phi)

    return x, y


# def ell_points(x, coeffs):
#     x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)

#     A = (cp.sin(phi) / ap) ** 2 + (cp.cos(phi) / bp) ** 2
#     B = -(2 * y0 * A)
#     C = y0**2 * A + (x - x0) ** 2 / ap ** 2 * (cp.cos(phi)) ** 2 - (x - x0) ** 2 / bp ** 2 * (cp.sin(phi)) ** 2 - 1

#     return (-B - cp.sqrt(B ** 2 - 4 * A * C)) / (2 * A)


def dellipse(
    x: float, y: float, a: float, b: float, c: float, d: float, f: float
) -> float:
    """
    The implicit derivative of the fitted ellipse at a point (x,y). This returns
    the gradient of the tangent to that point.

    Parameters
    ----------
    x : float
        The x-coordinate of the point to find the gradient of the tangent at.
    y : float
        The y-coordinate of the point to find the gradient of the tangent at.
    a : float
        The a parameter of the Cartesian ellipse equation.
    b : float
        The b parameter of the Cartesian ellipse equation * 2.
    c : float
        The c parameter of the Cartesian ellipse equation.
    d : float
        The d parameter of the Cartesian ellipse equation * 2.
    f : float
        The f parameter of the Cartesian ellipse equation * 2.

    Returns
    -------
     : float
        The gradient of the tangent of the ellipse at the point (x,y).
    """

    return -(2 * a * x + d + b * y) / (f + 2 * c * y + b * x)


def contact_angle(
    frame: torch.tensor, tab_grad: float, tab_intercept: float, device: Union[torch.device, str, int]
) -> Tuple[float, float, float]:
    """
    This function extract the edge of the droplet, fits an ellipse and
    calculates the left, right and average contact angle.

    Parameters
    ----------
    frame : torch.tensor
        The segmentation mask to fit the contact angle to.
    tab_grad : float
        The gradient of the tablet surface to calculate the contact angle with
        respect to.
    tab_intercept : float
        The intercept of the tablet surface.

    Returns
    -------
    phiL : float
        The contact angle from the left-side of the droplet.
    phiR : float
        The contact angle from the right-side of the droplet.
    phiAvg : float
        The contact angle average.
    """
    drop_img = frame

    y_edge, x_edge = torch.where(torch.diff(drop_img, axis=-2) == 1)
    edge = torch.stack((y_edge, x_edge), axis=1)
    edge = edge[edge[:, 0].argsort()]
    coeffs = fit_ellipse(edge[:, 1], edge[:, 0], device=device)
    x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
    A = bp**2 * (torch.cos(phi) + tab_grad*torch.sin(phi))**2 + ap**2 * (torch.sin(phi) - tab_grad*torch.cos(phi))**2
    B = (torch.sin(phi))**2 * (bp**2 * tab_grad * (tab_intercept - y0) - ap**2 * x0) + torch.sin(phi) * torch.cos(phi) * (ap**2 * (tab_grad*x0 + y0 - tab_intercept) + bp**2 * (tab_intercept - y0 - tab_grad*x0)) + (torch.cos(phi))**2 * (ap**2 * tab_grad * (tab_intercept - y0) - bp**2 * x0)
    B = 2*B
    C = (torch.cos(phi))**2 * (bp**2 * x0**2 + ap**2 * (tab_intercept - y0)**2) + torch.sin(phi) * torch.cos(phi) * (2 * x0 * bp**2 * (y0 - tab_intercept) + 2 * x0 * ap**2 * (tab_intercept - y0)) + (torch.sin(phi))**2 * (ap**2 * x0**2 + bp**2 * (tab_intercept - y0)**2) - ap**2 * bp**2
    xmax = (-1*B + torch.sqrt(B**2 - 4*A*C))/(2*A)
    xmin = (-1*B - torch.sqrt(B**2 - 4*A*C))/(2*A)
    ymax = tab_grad*xmax + tab_intercept
    ymin = tab_grad*xmin + tab_intercept

    mR = dellipse(xmax, ymax, *coeffs[:-1])
    mL = dellipse(xmin, ymin, *coeffs[:-1])

    phiR = torch.rad2deg(torch.arctan(torch.abs((tab_grad - mR) / (1 + tab_grad * mR))))
    phiL = torch.rad2deg(torch.arctan(torch.abs((tab_grad - mL) / (1 + tab_grad * mL))))
    phiAvg = (phiL + phiR) / 2

    return phiL, phiR, phiAvg
