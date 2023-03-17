import numpy as np
from math import sin, cos, sqrt

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

def rad2hms(rad : 'float') -> 'list[float]':
    '''
    ### Description:

    Function 

    ### Arguments



    ### Output

    
    '''
    degrees = np.rad2deg(rad)
    hours = degrees/15
    hoursInt = int(hours)
    min = int((hours-hoursInt) * 60)
    sec = ((hours-hoursInt) * 60 - min) * 60
    dms = [hoursInt, min, sec]
    return dms

def rad2dms(rad : 'float') -> 'Dms':
    dd = np.rad2deg(rad)
    dd = dd
    deg = int(np.trunc(dd))
    mnt = int(np.trunc((dd-deg) * 60))
    sec = ((dd-deg) * 60 - mnt) * 60
    dms = Dms(deg, abs(mnt), abs(sec))
    return dms

def dms2hms(dms : 'Dms') -> 'Hms':
    sall = dms.deg * (4*60) + dms.min * 4 + dms.sec/15    
    h = int(sall//3600)
    m = int((sall%3600)//60)
    s = sall%60
    return Hms(h, m, s)

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
    calculation of Greenwich Mean Sidereal Time - GMST in hours
    ----------
    jd : TYPE
        julian date
    '''
    T = (jd - 2451545) / 36525
    Tu = jd - 2451545
    g = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T**2-T**3/38710000
    g = (g%360) / 15
    return g

print(rad2hms(1))