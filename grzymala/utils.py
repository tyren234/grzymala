from math import sin, cos, sqrt, atan, tan
import numpy as np
import math

'''
    ### Description:

    ### Arguments


    ### Output
    
    '''

def blh2xyz(latitudeRad : 'float', longitudeRad : 'float', heightMeters : 'float') -> 'list[float]':
    '''
    ### Description:

    Function calculates x, y and z coordinates of a given point

    ### Arguments

    - latitudeRad : 'float' - point's latitude passed in radians
    - longitudeRad : 'float' - point's longitude passed in radians
    - heightMeters : 'float' - point's height in meters

    ### Output

    list[float] - list of three coordinates [x, y, z] representing the position of the given point
    '''
    a = 6378137 # meters
    eSquared = 0.00669438002290 # eccentricity squared 
    N = a / sqrt(1-(eSquared*(sin(latitudeRad)**2)))

    coords = [
        (N + heightMeters) * cos(latitudeRad) * cos(longitudeRad),
        (N + heightMeters) * cos(latitudeRad) * sin(longitudeRad),
        (N * (1 - eSquared) + heightMeters) * sin(latitudeRad)
    ]
    return coords

def xyz2phiLamH(objectXYZ : 'list[float]', a = 6378137,e2 = 0.00669438002290) -> 'tuple[float]':
    '''
    ### Description:

    Function calculates latitude, longitude and height of a given point

    ### Arguments

    - objectXYZ : 'list[float]' - object's x, y and z coordinates
    - optional: a : 'float'
    - optional: e2 : 'float'

    ### Output

    tuple[float] - tuple of three coordinates [phi, lambda, h] representing the position of the given point
    '''
    x,y,z = objectXYZ

    p = np.sqrt(x**2 + y**2)
    phi = np.arctan(z/(p*(1-e2)))
    
    while True:
        phi_stare = phi
        N = a / sqrt(1-(e2*(sin(phi)**2)))
        h = p/np.cos(phi) - N
        phi = np.arctan(z/(p * (1-(N*e2)/(N-h))))

        if abs(phi-phi_stare) < (0.000001/206265):
            break

    N = a / sqrt(1-(e2*(sin(phi)**2)))
    h = p/np.cos(phi) - N 
    lam = np.arctan2(y,x)

    return (phi, lam, h)

def Np(B: 'float', a : 'int' = 6378137, e2 : 'float' = 0.00669438002290) -> 'float':
    # TODO: description
    N = a/(1- e2*(np.sin(B)**2))**0.5
    return N

def hirvonen(X, Y, Z, a : 'int' = 6378137, e2 : 'float' = 0.00669438002290) -> 'tuple[float]':
    '''
    ### Description:

    Function calculates latitude, longitude and height of a given point

    ### Arguments

    - objectXYZ : 'list[float]' - object's x, y and z coordinates
    - optional: a : 'float'
    - optional: e2 : 'float'

    ### Output

    tuple[float] - tuple of three coordinates [B, L, H] representing the position of the given point
    '''
    r = (X**2 + Y**2)**0.5
    B = atan(Z/(r*(1-e2)))
    
    while 1:
        N = Np(B, a, e2)
        H = r/np.cos(B) - N
        Bst = B
        B = atan(Z/(r*(1-(e2*(N/(N+H))))))    
        if abs(Bst-B)<(0.00001/206265):
            break
    L = atan(Y/X)
    N = Np(B, a, e2)
    H = r/np.cos(B) - N
    return B, L, H 

def dms2deg(dms : 'list[float]') -> 'float':
    '''
    ### Description:

    Changes angle form degrees, minutes and seconds to float degrees.

    ### Arguments

    - dms: 'list[float]' - list consisting of [degrees, minutes and seconds]

    ### Output

    float value of angle in degrees

    '''
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    
    deg = degrees + minutes / 60 + seconds / 3600
    return deg

def hms2deg(hms : 'list[float]') -> 'float':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    degreesInHour = 15
    hours = hms[0]
    minutes = hms[1]
    seconds = hms[2]
    
    deg = hours * degreesInHour + minutes / 60 + seconds / 3600
    return deg

def deg2dms(degrees : 'float') -> 'list[float]':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    deg = int(degrees)
    mnt = int((degrees-deg) * 60)
    sec = ((degrees-deg) * 60 - mnt) * 60

    return [deg, abs(mnt), abs(sec)]

def deg2hms(degrees : 'float') -> 'list[float]':
    degreesInHour = 15
    hours = degrees / degreesInHour
    hoursInt = int(hours)
    min = int((hours-hoursInt) * 60)
    sec = ((hours-hoursInt) * 60 - min) * 60
    hms = [hoursInt, min, sec]
    return hms

# radians

def dms2rad(dms : 'list[float]') -> 'float':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    
    d = dms[0]
    m = dms[1]
    s = dms[2]
    
    deg = d + m / 60 + s / 3600
    rad = np.deg2rad(deg)
    return rad

def rad2hms(rad : 'float') -> 'list[float]':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    degrees = np.rad2deg(rad)
    hms = deg2hms(degrees)
    return hms

def rad2dms(rad : 'float') -> 'list[float]':
    degrees = np.rad2deg(rad)
    dms = deg2dms(degrees)
    return dms

def hms2rad(hms : 'list[float]') -> 'float':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    h = hms[0]
    m = hms[1]
    s = hms[2]
    
    deg = h * 15 + m / 60 + s / 3600
    rad = np.deg2rad (deg * 15)
    return rad

# other

def dms2hms(dms : 'list[float]') -> 'list[float]':
    degreesInHour = 15
    d, m, s = dms
    return [int(d / degreesInHour), m, s]

def hms2sec(hms : 'list[float]') -> 'float':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    sec = hms[0] * 3600 + hms[1] * 60 + hms[2]
    return sec

def sec2hms(seconds : 'float') -> 'list[float]':
    '''
    ### Description:

    Function converts seconds to hours, minutes, seconds.

    ### Arguments



    ### Output

    
    '''
    h = seconds // 3600
    m = (seconds - (h * 3600)) // 60
    s = seconds - h * 3600 - m * 60

    return [h, m, s]

def GaussKruger2PL92(xGK : 'float', yGK : 'float', m92 = 0.9993):
    '''
    ### Description:

    Function converts Gauss-Kruger coordinates to PL-1992 projection.

    ### Arguments
    xGK : local X coordinate converted to Gauss-Kruger system
    yGK : local Y coordinate converted to Gauss-Kruger system

    ### Output
    x92 : local X coordinate converted to PL-1992 system
    y92 : local Y coordinate converted to PL-1992 system
    
    '''
    x92 = m92 * xGK - 5300000
    y92 = m92 * yGK + 500000

    return x92, y92

def GaussKruger2PL00(xGK : 'float', yGK : 'float', zoneNumber : 'float', m00 = 0.999923):
    '''
    ### Description:

    Function converts Gauss-Kruger coordinates to PL-2000 projection.

    ### Arguments
    xGK : local X coordinate converted to Gauss-Kruger system
    yGK : local Y coordinate converted to Gauss-Kruger system

    ### Output
    x92 : local X coordinate converted to PL-2000 system
    y92 : local Y coordinate converted to PL-2000 system
    
    '''
    x00 = m00 * xGK 
    y00 = m00 * yGK + zoneNumber * 1000000 + 500000 

    return x00, y00

def blh2GaussKruger(latDeg : 'float', lonDeg : 'float', zoneLongtitude: 'float', a = 6378137, eSquared = 0.00669438002290):
    '''
    ### Description:

    Function converts BLH coordinates to Gauss-Kruger's projection coordinates. 

    ### Arguments

    latDeg : latitude of the point [DEGREES]
    lonDeg : longtitude of the point [DEGREES]
    zoneLongtitude : longtitude of the specific zone in Gauss-Kruger projection [DEGREES]

    ### Output
    xGK : local X coordinate converted to Gauss-Kruger system
    yGK : local Y coordinate converted to Gauss-Kruger system

    '''
    phi = np.deg2rad(latDeg)
    lam = np.deg2rad(lonDeg)
    zone = np.deg2rad(zoneLongtitude)

    # elipsoidal params
    bSquared = (a**2) * (1 - eSquared)
    eSquaredPrim = ((a**2) - bSquared) / bSquared

    delta_lam = lam - zone
    t = tan(phi)
    etaSquared = eSquaredPrim * (cos(phi)**2)
    N = a / (sqrt(1 - eSquared * np.power(np.sin(phi), 2)))

    # length of the longtitude's arc
    A_0 = 1 - (eSquared / 4) - (3 * eSquared**2 / 64) - (5 * eSquared**3) / 256
    A_2 = (3 / 8) * (eSquared + (eSquared**2 / 4) + (15 * eSquared**3 / 128))
    A_4 = (15 / 256) * (eSquared**2 + (3 * eSquared**3 / 4))
    A_6 = (35 * eSquared**3) / 3072
    rho = a * (A_0 * phi - A_2 * sin(2 * phi) + A_4 * sin(4 * phi) + A_6 * sin(6 * phi))

    # Gauss-Kruger local coordinates
    xGK = rho + (delta_lam**2 / 2) * N * sin(phi) * cos(phi) * (1 + (delta_lam**2 / 12) * (cos(phi)**2) * (5 - t**2 + 9 * etaSquared + 4 * etaSquared**2)
                                + ((delta_lam**2 / 360) * (cos(phi)**4) * (61 - 58 * t**2 + t**4 + 270 * etaSquared - 330 * etaSquared * t**2)))
    yGK = delta_lam * N * cos(phi) * (1 + ((delta_lam**2 / 6) * (cos(phi)**2) * (1 - t**2 + etaSquared)) +
                                    ((delta_lam**4 / 120) * (cos(phi)**4) * (5 - 18 * t**2 + t**4 + 14 * etaSquared - 58 * etaSquared * t**2)))

    return xGK, yGK

def GaussKruger2blh(xGK : 'float', yGK : 'float', zoneLongtitude : 'float', a = 6378137, eSquared = 0.00669438002290):
    '''
    ### Description:

    Function converts Gauss-Kruger's projection coordinates to latitude and longtitude. 

    ### Arguments
    xGK : local X coordinate converted to Gauss-Kruger system
    yGK : local Y coordinate converted to Gauss-Kruger system

    ### Output
    latDeg : latitude of the point [DEGREES]
    lonDeg : longtitude of the point [DEGREES]

    
    '''
    A0 = 1 - eSquared/4 - (3 * eSquared**2) / 64 - (5 * eSquared**3) / 256
    A2 = (3 / 8) * (eSquared + (eSquared**2) / 4 + (15 * eSquared**3) / 128)
    A4 = (15 / 256) * (eSquared**2 + (3 * eSquared**3) / 4)
    A6 = (35 * eSquared**3) / 3072
    bSquared = (a**2) * (1 - eSquared)
    eSquaredPrim = ((a**2) - bSquared) / bSquared
    phiTemp = xGK / (a * A0)
    
    while 1:
        dl = a * (A0 * phiTemp - A2 * sin(2 * phiTemp) + A4 * sin(4 * phiTemp) - A6 * sin(6 * phiTemp))
        phiTemp_n = phiTemp + (xGK - dl) / (a * A0)
        dl_n = a * (A0 * phiTemp_n - A2 * sin(2 * phiTemp_n) + A4 * sin(4 * phiTemp_n) - A6 * sin(6 * phiTemp_n))
        if (abs(phiTemp_n - phiTemp) <= np.radians(0.000001 / 3600)):
            phiTemp = phiTemp_n
            dl = dl_n
            break
        else:
            phiTemp = phiTemp_n
            dl = dl_n
    N1 = a / sqrt(1 - eSquared * ((sin(phiTemp)) ** 2))
    M1 = (a * (1 - eSquared)) / sqrt((1 - eSquared * (sin(phiTemp) ** 2)) ** 3)
    t1 = tan(phiTemp)
    n12 = eSquaredPrim * ((cos(phiTemp)) ** 2)

    latDeg = phiTemp - yGK ** 2 * t1 / (2 * M1 * N1) * (1 - yGK ** 2 / (12 * N1 ** 2) * (5 + 3 * t1 ** 2 + n12 - 9 * n12 * t1 ** 2 - 4 * n12 ** 4) + yGK ** 4 / (360 * N1 ** 4) * (61 + 90 * t1 ** 2 + 45 * t1 ** 4))
    lonDeg = zoneLongtitude + yGK / (N1 * cos(phiTemp)) * (1 - (yGK ** 2 / (6 * N1 ** 2)) * (1 + 2 * t1 ** 2 + n12) + ((yGK ** 4) / (120 * N1 ** 4)) * (5 + 28 * t1 ** 2 + 24 * t1 ** 4 + 6 * n12 + 8 * n12 * t1 ** 2))

    return np.degrees(latDeg), np.degrees(lonDeg)

def gaussArea(x : 'np.array', y : 'np.array'):
    '''
    ### Description:

    Function calculates area inside given points. 

    ### Arguments
    x : array of X coordinates
    y : array of Y coordinates

    ### Output
    area : calculated area
    
    '''
    if len(x) < 3:
        raise Exception("Number of points should be greater than 2!")
    if len(x) != len(y):
        raise Exception("Arrays should be the same length!")
    
    doubleArea = 0
    for i in range(len(x)):
        if i != len(x) - 1:
            doubleArea += x[i] * (y[i + 1] - y[i - 1])
        else:
            doubleArea += x[i] * (y[0] - y[i - 1])
    area = doubleArea / 2

    return area

def gaussAreaCheck(x : 'np.array', y : 'np.array'):
    '''
    ### Description:

    Function calculates area inside given points.

    ### Arguments
    x : array of X coordinates
    Y : array of Y coordinates

    ### Output
    area : calculated area
    
    '''
    if len(x) < 3:
        raise Exception("Number of points should be greater than 2!")
    if len(x) != len(y):
        raise Exception("Arrays should be the same length!")
    
    doubleArea = 0
    for i in range (len(y)):
        if i != len(y) - 1:
            doubleArea += y[i] * (x[i + 1] - x[i - 1])
        else:
            doubleArea += y[i] * (x[0] - x[i - 1])
    area = doubleArea / -2

    return area

def julday(y : 'int', m : 'int', d : 'int', h : 'int') -> 'int':
    '''
    ### Description:

    Gives simplified Julian date. Valid only between 1 March 1900 to 28 February 2100

    ### Arguments
    y : year between 1900 and 2100
    m : month (0-11)
    d : day of the monh (1-31)
    h : current hour

    ### Output
    date : calculated julian date
    
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
        
    date = np.floor(365.25 * (y + 4716)) + np.floor(30.6001 * (m + 1)) + d + h / 24 - 1537.5
    return date

def GMST(julianDate : 'int') -> 'float':
    '''
    Returns Greenwich Mean Sidereal Time - GMST in hours
    
    julianDate : see julday(y : 'int', m : 'int', d : 'int', h : 'int') -> 'int':
    '''
    T = (julianDate - 2451545) / 36525
    Tu = julianDate - 2451545
    g = 280.46061837 + 360.98564736629*(julianDate - 2451545.0) + 0.000387933*T**2-T**3/38710000
    g = (g%360) / 15
    return g

def kivioja(phiDegrees : 'float', lamDegrees : 'float', azimuthDegrees : 'float', s, 
            a = 6378137, e = 0.00669438002290):
    phi = np.deg2rad(phiDegrees)
    lam = np.deg2rad(lamDegrees)
    az = np.deg2rad(azimuthDegrees)

    n = int(math.ceil(s/1.5))
    delta_s = np.empty(n)
    for i in range(0, n-1):
        delta_s[i] = 1.5
    delta_s[-1] = s - (n-1)*1.5

    a = 6378137
    e = 0.00669438002290

    N = 0
    M = 0
    delta_phi, delta_lam = [0, 0]
    for i in range(0, n):
        N = a / (np.sqrt(1 - e * np.power(np.sin(phi), 2)))
        M = a*(1-e)/(np.sqrt(np.power(1-e*np.sin(phi)**2, 3)))

        delta_phi = delta_s[i]*np.cos(az)/M
        delta_az = delta_s[i]*np.sin(az)*np.tan(phi)/N

        phi_m = phi + (1/2)*delta_phi
        az_m = az + (1/2)*delta_az

        Nm = a / (np.sqrt(1 - e * np.power(np.sin(phi_m), 2)))
        Mm = a * (1 - e) / (np.sqrt(np.power(1 - e * np.sin(phi_m) ** 2, 3)))

        delta_phi_m = (delta_s[i] * np.cos(az_m)) / Mm
        delta_lam_m = (delta_s[i] * np.sin(az_m)) / (Nm * np.cos(phi_m))
        delta_az_m = (delta_s[i] * np.sin(az_m) * np.tan(phi_m)) / Nm

        phi = phi + delta_phi_m
        lam = lam + delta_lam_m
        az = az + delta_az_m

    return [np.rad2deg(phi), np.rad2deg(lam), np.rad2deg(az)]

def vincenty(latA : 'float', lonA : 'float', latB : 'float', lonB : 'float', a = 6378137, eSquared = 0.00669438002290):
    '''
    Parameters
    ----------
    latA : latitude of point A [RAD]
    lonA : longtitude of point A [RAD]
    latB : latitude of point B [RAD]
    lonB : longtitude of point B [RAD]

    Returns
    -------
    sAB : length of the geodetic line between points A and B [METERS]
    Az_AB : azimuth of the geodetic line between points A and B [RAD]
    Az_BA : reverse azimuth of the geodetic line between points A and B [RAD]
    '''
    b = a * sqrt(1 - eSquared)
    f = 1 - b / a
    dL = lonB - lonA
    UA = atan((1 - f) * tan(latA))
    UB = atan((1 - f) * tan(latB))
    L = dL

    while True:
        sin_sig = sqrt((cos(UB) * sin(L))**2 + (cos(UA) * sin(UB) - sin(UA) *  cos(UB) * cos(L))**2)
        cos_sig = sin(UA) * sin(UB) + cos(UA) * cos(UB) * cos(L)
        sig = np.arctan2(sin_sig, cos_sig)

        sin_al = (cos(UA) * cos(UB) *  sin(L)) / sin_sig
        cos2_al = 1 - sin_al**2
        cos2_sigm = cos_sig - (2 * sin(UA) * sin(UB)) / cos2_al
        C = (f / 16) * cos2_al * (4 + f * (4 - 3 * cos2_al))
        Lst = L
        L = dL + (1-C) * f * sin_al * (sig + C * sin_sig * (cos2_sigm + C * cos_sig * (-1 + 2 * cos2_sigm**2)))
        if abs(L - Lst) < (0.000001 / 206265):
            break
    
    u2 = (a**2 - b**2) / (b**2) * cos2_al
    A = 1 + (u2/16384) * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    d_sig = B * sin_sig * (cos2_sigm + 1 / 4 * B * (cos_sig * (-1 + 2 * cos2_sigm**2)\
            - 1 / 6 * B * cos2_sigm * (-3 + 4 * sin_sig**2)*(-3 + 4 * cos2_sigm**2)))
    sAB = b * A * (sig - d_sig)
    
    Az_AB = np.arctan2((cos(UB) * sin(L)), (cos(UA) * sin(UB) - sin(UA) * cos(UB) * cos(L)))
    Az_BA = np.arctan2((cos(UA) * sin(L)), (-sin(UA) * cos(UB) + cos(UA) * sin(UB) * cos(L))) + np.pi

    return sAB, Az_AB, Az_BA


def countNEU(ovserverLatitudeRad : 'float', observerLongitudeRad : 'float'):
    n = np.array([
        -math.sin(ovserverLatitudeRad)*math.cos(observerLongitudeRad),
        -math.sin(ovserverLatitudeRad)*math.sin(observerLongitudeRad),
        math.cos(ovserverLatitudeRad)
    ])

    e = np.array([
        -math.sin(observerLongitudeRad),
        math.cos(observerLongitudeRad),
        0
    ])

    u = np.array([
        math.cos(ovserverLatitudeRad)*math.cos(observerLongitudeRad),
        math.cos(ovserverLatitudeRad)*math.sin(observerLongitudeRad),
        math.sin(ovserverLatitudeRad)
    ])

    return [n,e,u]

def countRneut(n,e,u):
    Rneu = np.column_stack((n,e,u))
    Rneut = Rneu.transpose()
    
    return Rneut

def countNeuVector(Rneut, vectorObjectObserver):
    neuVector = Rneut @ vectorObjectObserver

    return neuVector

def countDistance(neuVector):
    s = np.sqrt(np.square(neuVector[0]) + np.square(neuVector[1]) + np.square(neuVector[2]))
    return s

def countAz(neuVectorE, neuVectorN):
    Azs = np.arctan2(neuVectorE,neuVectorN)
    if(Azs < 0): Azs += 2 * math.pi

    return Azs

def countHs(neuVector):
    hs = np.arcsin(neuVector[2]/(np.sqrt(np.square(neuVector[0]) + np.square(neuVector[1]) + np.square(neuVector[2]))))

    return hs

def countGDOP(QMatrix : 'np.array'):
    if QMatrix.shape != (4,4):
        raise ValueError
    return sqrt(QMatrix.trace())

def countTDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (4, 4):
        raise ValueError
    return sqrt(QneuMatrix[3,3])

def countHDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (3,3):
        raise ValueError
    return sqrt(QneuMatrix[0,0] + QneuMatrix[1,1])

def countVDOP(QneuMatrix : 'np.array'):
    if QneuMatrix.shape != (3,3):
        raise ValueError
    return sqrt(QneuMatrix[2,2])

def countPDOP(QMatrix : 'np.array'):
    if QMatrix.shape != (3,3) and QMatrix.shape != (4,4):
        raise ValueError
    return sqrt(QMatrix[0,0] + QMatrix[1,1] + QMatrix[2,2])

def countVectorObserverObject (objectsCoords : 'np.array', observersCoords : 'np.array'):
    vectorObserverObject = objectsCoords - observersCoords
    return vectorObserverObject

def countElevation(neuVector):
    hs = np.arcsin(neuVector[2]/(np.sqrt(np.square(neuVector[0]) + np.square(neuVector[1]) + np.square(neuVector[2]))))
    return hs

def countElevationAndAzimtuhDistance(objectXYZ : 'np.array', observerXYZ : 'np.array'):
    observersPhiLamH = xyz2phiLamH(observerXYZ)
    vectorObjectObserver = countVectorObserverObject(objectXYZ, observerXYZ)
    n, e, u = countNEU(observersPhiLamH[0], observersPhiLamH[1])
    Rneut = countRneut(n, e, u)
    vectorNeu = countNeuVector(Rneut, vectorObjectObserver)
    elevation, azimuth, distance = countElevation(vectorNeu), countAz(vectorNeu[1], vectorNeu[0]), countDistance(vectorNeu)
    return [elevation, azimuth, distance]