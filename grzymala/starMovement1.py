import numpy as np
import matplotlib.pyplot as plt
from utils import *
from loader import LoadingAnimation

def time2UTC0(time : 'float', difference: 'int'):
    return time - difference

def calculateLocalStarTime (year, month, day, hour, longitude) -> 'float':
    julianDate = julday(year, month, day, hour)
    gmst0 = GMST(julianDate)

    roundAngle = 360
    numberOfDegreesInHour = 15

    localStarTime = gmst0 * numberOfDegreesInHour + longitude
    if localStarTime > roundAngle:
        localStarTime -= roundAngle

    localStarTime = np.deg2rad(localStarTime)

    return localStarTime

def calculateHourAngle(localStarTime : 'float', rightAscension : 'float') -> 'float':
    return localStarTime - rightAscension

def calculateAzimuthAndHeight(phiRad : 'float', deltaRad : 'float', tRad : 'float') -> 'list[float, float]':
    hRad = math.asin((math.sin(phiRad) * math.sin(deltaRad)) + (math.cos(phiRad) * math.cos(deltaRad) * math.cos(tRad)))

    AzRad = math.atan2(((-1) * math.cos(deltaRad) * math.sin(tRad)),
                        ((math.cos(phiRad) * math.sin(deltaRad)) - (math.sin(phiRad) * math.cos(deltaRad) * math.cos(tRad))))
    if AzRad < 0: AzRad += (2 * math.pi)

    return [AzRad, hRad]

def starMovement(observerLongitudeDeg: 'float', 
                observerLatitudeDeg: 'float',
                year: 'int',
                month: 'int',
                day: 'int',
                starRightAscensionHms: 'list[float]',
                starDeclinationDms: 'list[float]', 
                timezoneTimeDifference: 'int') -> 'list[tuple[float, float]]':
    
    loading = LoadingAnimation(finishMessage = "Done!")
    loading.start()

    hoursRange = range(24)

    AzHPairsObserver = []

    for hour in hoursRange:
        utc = time2UTC0(hour, timezoneTimeDifference)
        
        # obliczenie LST w radianach
        lst = calculateLocalStarTime(year, month, day, utc, observerLongitudeDeg)
        if utc < 0:
            lst = calculateLocalStarTime(year,6,30, utc, observerLongitudeDeg)

        phi = np.deg2rad(observerLatitudeDeg)
        delta = dms2rad(starDeclinationDms)
        
        t = calculateHourAngle(lst, hms2rad(starRightAscensionHms))

        AzHPairsObserver.append(calculateAzimuthAndHeight(phi, delta, t))

    loading.stop()
    return AzHPairsObserver

def skyplot(figure: 'matplotlib.pyplot.figure', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False) -> 'matplotlib.pyplot.figure':
    ax = figure.add_subplot(polar = True)
    ax.set_theta_zero_location('N') # ustawienie kierunku północy na górze wykresu
    ax.set_theta_direction(-1)

    ax.set_yticks(range(0, 90+10, 10)) # Define the yticks

    yLabel = ['90', '', '', '60', '', '', '30', '', '', '']
    ax.set_yticklabels(yLabel)
    ax.set_rlim(0,90)

    xs = [x[0] for x in AzHPairsObserver]
    ys = [90-np.rad2deg(x[1]) for x in AzHPairsObserver]
    ax.scatter(xs, ys, label="Observer")

    if legend:
        ax.legend()
    return figure

def heightAzimuth (figure: 'matplotlib.pyplot.figure', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False):
    ax = figure.add_subplot()
    xs = [x[0] for x in AzHPairsObserver]
    ys = [x[1] for x in AzHPairsObserver]
    ax.scatter(xs, ys, color="red", label="Observer")

    move = 0.02
    for x,y in AzHPairsObserver:
        ax.annotate(str(round(y,2)),(x+move, y+move))

    ax.set_title("Height to azimuth")
    ax.set_xlabel("Azimuth Az (radians)")
    ax.set_ylabel("Haight h (radians)")
    ax.grid();
    if legend:
        ax.legend()
    return figure
