import numpy as np
import matplotlib.pyplot as plt
from utils import *

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

    return AzHPairsObserver

def skyplot(subplot: 'matplotlib.axes(polar = True)', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False) -> 'matplotlib.axes':
    subplot.set_theta_zero_location('N')
    subplot.set_theta_direction(-1)

    subplot.set_yticks(range(0, 90+10, 10))

    yLabel = ['90', '', '', '60', '', '', '30', '', '', '0']
    subplot.set_yticklabels(yLabel)
    subplot.set_rlim(0,90)

    xs = [x[0] for x in AzHPairsObserver]
    ys = [90-np.rad2deg(x[1]) for x in AzHPairsObserver]
    subplot.scatter(xs, ys, label="Observer")
    subplot.set_title("Skyplot")

    if legend:
        subplot.legend()
    return subplot

def heightAzimuth (subplot: 'matplotlib.axes', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False) -> 'matplotlib.axes':
    xs = [x[0] for x in AzHPairsObserver]
    ys = [x[1] for x in AzHPairsObserver]
    subplot.scatter(xs, ys, label="Observer")

    move = 0.02
    for x,y in AzHPairsObserver:
        subplot.annotate(str(round(y,2)),(x+move, y+move))

    subplot.set_title("Relationship between Height and Azimuth")
    subplot.set_xlabel("Azimuth Az (radians)")
    subplot.set_ylabel("Haight h (radians)")
    # subplot.grid()
    if legend:
        subplot.legend()
    return subplot

def heightTime (subplot: 'matplotlib.axes', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False) -> 'matplotlib.axes':
    hoursRange = range(24)
    xs = [x for x in hoursRange]
    ys = [x[1] for x in AzHPairsObserver]
    subplot.scatter(xs, ys)
    subplot.plot(xs, ys, label="Observer")

    move = 0.02
    for i, x in enumerate(hoursRange):
        subplot.annotate(str(round(ys[i],2)),(x+move, ys[i]+move))

    subplot.set_title("Relationship between Height and Time")
    subplot.set_xlabel("Hour")
    subplot.set_ylabel("Height h (radians)")
    subplot.set_xticks(hoursRange)
    numberOfVerticalTicks = 8
    move = np.mean(ys) / numberOfVerticalTicks
    subplot.set_yticks([x for x in np.arange(min(ys) - move, max(ys) + move, ((max(ys) - min(ys)) / numberOfVerticalTicks))])
    subplot.grid()
    if legend:
        subplot.legend()
    return subplot

def spherePlot (subplot: 'matplotlib.axes(projection = "3d")', AzHPairsObserver : 'list[tuple[float, float]]', legend: bool = False) -> 'matplotlib.axes':
    # plt.rcParams["figure.figsize"] = [10.00, 10.00]
    # plt.rcParams["figure.autolayout"] = True
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    r = 1
    u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : 2 * np.pi : 20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # z[z<0] = 0
    subplot.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.1)

    # linie
    start = 2
    gx = [r * np.sin(x[0]) * np.cos(x[1]) for x in AzHPairsObserver[start:]]
    gy = [r * np.cos(x[1]) * np.cos(x[0]) for x in AzHPairsObserver[start:]]
    gz = [r * np.sin(x[1]) for x in AzHPairsObserver[start:]]
    subplot.plot3D(gx, gy, gz, label="Observer")

    # punkty
    subplot.scatter3D(gx[0], gy[0], gz[0])
    subplot.scatter3D(gx[-1], gy[-1], gz[-1])

    subplot.set_title("Star movement on the sphere")
    if legend:
        subplot.legend()

    return subplot

