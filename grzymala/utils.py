import numpy as np
from math import sin, cos, sqrt, atan
import math

'''
    ### Description:

    Function 

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

def xyz2phiLamH(objectXYZ : 'list[float]', a = 6378137,e2 = 0.00669438002290):
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

def Np(B, a : 'int' = 6378137, e2 : 'float' = 0.00669438002290):
    N = a/(1- e2*(np.sin(B)**2))**0.5
    return N

def hirvonen(X, Y, Z, a : 'int' = 6378137, e2 : 'float' = 0.00669438002290):
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

print(xyz2phiLamH([12,12,12]), hirvonen(12,12,12))
# degrees

def dms2deg(dms : 'list[float]') -> 'float':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
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

    Function 

    ### Arguments



    ### Output

    
    '''
    h = seconds // 3600
    m = (seconds - (h * 3600)) // 60
    s = seconds - h * 3600 - m * 60

    return [h, m, s]

def julday(y,m,d,h):
    '''
    Simplified Julian Date generator, valid only between
    1 March 1900 to 28 February 2100
    '''
    if m <= 2:
        y = y - 1
        m = m + 12
        
    jd = np.floor(365.25*(y+4716))+np.floor(30.6001*(m+1))+d+h/24-1537.5;
    return jd

def GMST(jd):
    '''
    Greenwich Mean Sidereal Time - GMST in hours
    
    jd : julian date - see julday(y,m,d,h)
    '''
    T = (jd - 2451545) / 36525
    Tu = jd - 2451545
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2-T**3/38710000
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

    a = 6378137             # wielka półoś dla elipsoidy GRS80
    e = 0.00669438002290    # kwadrat pierwszego mimośrodu dla elipsoidy GRS80

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