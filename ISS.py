#http://celestrak.com/NORAD/elements/

import pathlib
from skyfield.api import load, wgs84
from pytz import timezone, common_timezones
import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shapereader
import cartopy.io.img_tiles as cimgt

ephem   = load('de421.bsp') #1900-2050 only
sun     = ephem['sun']
earth   = ephem['earth']
moon    = ephem['moon']

hokoon  = wgs84.latlon(22+23/60+1/3600, 114+6/60+29/3600)

ts      = load.timescale()
tz      = timezone('Asia/Hong_Kong')
t_now   = ts.now().astimezone(tz)

R_e = 6378.140 # earth radius in km

#####initial######
#####setting######
bg_object = 'SUN'
fg_object = 'MOON'

#t_range = ts.utc(t_now.year, t_now.month, t_now.day, -8, 0, range(0, 86400))
t_range = ts.utc(2035, 9, 2, 1.9234653, 0, range(-43200, 43200))
#t_range = ts.utc(2021, 4, 17, 0, 0, 0)
#t_range = ts.utc(2035, 9, 2, 1, 1, 21)
#print(ts.utc(2035, 9, 2, 0, 0, 0).delta_t)
#print(ts.utc(2035, 9, 2, 1.9234653, 0, 0).delta_t)
##################

ISS     = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=25544', reload=True)[0]
##SL26 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44240', reload=True)[0]
##SL61 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44249', reload=True)[0]
##SL71 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44252', reload=True)[0]
##SL43 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44257', reload=True)[0]
##SL64 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44275', reload=True)[0]
##SL68 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44279', reload=True)[0]
##SL70 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44281', reload=True)[0]
##SL75 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44286', reload=True)[0]
##SL76 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44287', reload=True)[0]
##SL48 = load.tle_file('https://celestrak.com/satcat/tle.php?CATNR=44289', reload=True)[0]

print('orbital elements epoch:\n'+str(ISS.epoch.astimezone(tz)))
print('{:.2f} hours after epoch'.format((ts.now()-ISS.epoch)*24))

ra, dec, edistance = (ISS).at(t_range).radec()
alt, az, tdistance = (ISS - hokoon).at(t_range).altaz()

# geocentric equatorial position vector of ISS
X_i, Y_i, Z_i = ISS.at(t_range).position.km/R_e
print(ISS.at(t_range).radec(epoch='date'))
# geocentric equatorial position vector of sun
X_s = earth.at(t_range).observe(sun).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(sun).apparent().radec(epoch='date')[1].degrees))\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(sun).apparent().radec(epoch='date')[0].hours*15))
Y_s = earth.at(t_range).observe(sun).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(sun).apparent().radec(epoch='date')[1].degrees))\
      *numpy.sin(numpy.radians(earth.at(t_range).observe(sun).apparent().radec(epoch='date')[0].hours*15))
Z_s = earth.at(t_range).observe(sun).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.sin(numpy.radians(earth.at(t_range).observe(sun).apparent().radec(epoch='date')[1].degrees))

# geocentric equatorial position vector of moon's CG
X_c = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[1].degrees))\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[0].hours*15))
Y_c = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.cos(numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[1].degrees))\
      *numpy.sin(numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[0].hours*15))
Z_c = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e\
      *numpy.sin(numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[1].degrees))

ra_c    = numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[0].hours*15)
dec_c   = numpy.radians(earth.at(t_range).observe(moon).apparent().radec(epoch='date')[1].degrees)

##print(earth.at(ts.utc(2035, 9, 2, 1, 0, 0)).observe(sun).apparent().radec(epoch='date')[0])
##print(earth.at(ts.utc(2035, 9, 2, 1, 0, 0)).observe(sun).apparent().radec(epoch='date')[1])
##print(earth.at(ts.utc(2035, 9, 2, 1, 0, 0)).observe(moon).apparent().radec(epoch='date')[0])
##print(earth.at(ts.utc(2035, 9, 2, 1, 0, 0)).observe(moon).apparent().radec(epoch='date')[1])

# moon geometrical correction
T_s         = (t_range-ts.utc(2000, 1, 1, 0, 0, 0)+t_range.delta_t)/36525
epsilon     = numpy.radians((23+26/60+21.448/3600)-(46.8150/3600)*T_s-(0.00059/3600)*T_s*T_s+(0.001813/3600)*T_s*T_s*T_s)

S_beta      = numpy.cos(epsilon)*numpy.sin(dec_c)-numpy.sin(epsilon)*numpy.cos(dec_c)*numpy.sin(ra_c)
C_beta      = numpy.sqrt(1-S_beta*S_beta)
S_tau       = numpy.sin(epsilon)*numpy.cos(ra_c)/C_beta
C_tau       = (numpy.cos(epsilon)-numpy.sin(dec_c)*S_beta)/(numpy.cos(dec_c)*C_beta)

delta_lon   = numpy.radians(0.5/3600)
delta_lat   = numpy.radians(-0.25/3600)

delta_ra    = (delta_lon*C_beta*C_tau-delta_lat*S_tau)/numpy.cos(dec_c)
delta_dec   = (delta_lon*C_beta*S_tau+delta_lat*C_tau)

ra_m        = ra_c+delta_ra
dec_m       = dec_c+delta_dec

X_m         = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e*numpy.cos(dec_m)*numpy.cos(ra_m)
Y_m         = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e*numpy.cos(dec_m)*numpy.sin(ra_m)
Z_m         = earth.at(t_range).observe(moon).apparent().radec(epoch='date')[2].km/R_e*numpy.sin(dec_m)

### Besselian elements ###
# geocentric equatorial RA Dec of shadow axis
if bg_object == 'SUN':
    if fg_object == 'ISS':
        print('ISS solar transit')
        X_rel   = X_s-X_i
        Y_rel   = Y_s-Y_i
        Z_rel   = Z_s-Z_i
    elif fg_object == 'MOON':
        print('solar eclipse')
        X_rel   = X_s-X_m
        Y_rel   = Y_s-Y_m
        Z_rel   = Z_s-Z_m
elif bg_object == 'MOON':
    if fg_object == 'ISS':
        print('ISS lunar transit')
        X_rel   = X_m-X_i
        Y_rel   = Y_m-Y_i
        Z_rel   = Z_m-Z_i
    else:
        print('WTF')

d       = numpy.arctan(Z_rel/numpy.hypot(X_rel,Y_rel))
S_d     = numpy.sin(d)
C_d     = numpy.cos(d)

a       = numpy.arctan2(Y_rel,X_rel)
S_a     = numpy.sin(a)
C_a     = numpy.cos(a)

# fundamental position vector of foreground object
if fg_object == 'ISS':
    x_fg     = -S_a*X_i+C_a*Y_i
    y_fg     = -S_d*C_a*X_i-S_d*S_a*Y_i+C_d*Z_i
    #z_fg    = C_d*C_a*X_i+C_d*S_a*Y_i+S_d*Z_i
elif fg_object == 'MOON':
    x_fg     = -S_a*X_m+C_a*Y_m
    y_fg     = -S_d*C_a*X_m-S_d*S_a*Y_m+C_d*Z_m
    #z_fg    = C_d*C_a*X_m+C_d*S_a*Y_m+S_d*Z_m

# Greenwich Hour angle mu
mu      = numpy.radians(15*t_range.gast)-a
S_mu    = numpy.sin(mu)
C_mu    = numpy.cos(mu)
#print(t_range.gast*15)
# project shadow on earth surface in fundamental frame (x_i,y_i,z_t) by eq(5.3)
ee      = 0.006694385
z_a     = -ee*y_fg*C_d*S_d
z_b     = (1-ee)*(1-x_fg*x_fg-y_fg*y_fg-ee*(1-x_fg*x_fg)*C_d*C_d)
z_c     = 1-ee*C_d*C_d

z_t     = (z_a+numpy.sqrt(z_b))/z_c

# transform to geocentric position vector on earth surface (u,v,w) by eq(5.5)
u       = x_fg*S_mu-y_fg*C_mu*S_d+z_t*C_mu*C_d
v       = x_fg*C_mu+y_fg*S_mu*S_d-z_t*S_mu*C_d
w       = y_fg*C_d+z_t*S_d

# tranform (u,v,w) to (lan,lon) by eq(5.8) & eq(5.14)
lon     = numpy.degrees(numpy.arctan2(v,u))

r       = numpy.sqrt(u*u+v*v+w*w)
uv      = u*u+v*v
lat_z   = w/uv
lat_a   = ee/r
lat_b   = ee*ee/(2*r*r*r)*(2*uv/r+w*w)
lat_c   = ee*ee*ee/(r*r*r*r*r)*(uv*(2*uv-3*w*w)/(2*r*r)+2*uv*w*w/r+3*w*w*w*w/8)
lat     = numpy.degrees(numpy.arctan(lat_z*(1+lat_a+lat_b+lat_c)))

### summary ###
print('################ Besselian elements #################')
print('r_s:')
print(earth.at(t_range).observe(sun).distance().au[43200])
print('ra_s:')
print(15*earth.at(t_range).observe(sun).radec()[0].hours[43200])
print('dec_s:')
print(earth.at(t_range).observe(sun).radec()[1].degrees[43200])
print('r_m:')
print(earth.at(t_range).observe(moon).distance().au[43200])
print('ra_m:')
print(15*earth.at(t_range).observe(moon).radec()[0].hours[43200])
print('dec_m:')
print(earth.at(t_range).observe(moon).radec()[1].degrees[43200])

print('(X_s, Y_s, Z_s):')
print(X_s[43200], Y_s[43200], Z_s[43200])
print('(X_m, Y_m, Z_m):')
print(X_m[43200], Y_m[43200], Z_m[43200])
print('(G_X,G_Y,G_Z):')
print(X_rel[43200], Y_rel[43200], Z_rel[43200])
print('a:')
print(numpy.degrees(a)[43200])
print('d:')
print(numpy.degrees(d)[43200])
print('g:')
print(numpy.sqrt(X_rel*X_rel+Y_rel*Y_rel+Z_rel*Z_rel)[43200])
print('(x_0,y_0):')
print(x_fg[43200],y_fg[43200])

print('mu:')
print(numpy.degrees(mu)[43200])
print('z:')
print(z_t[43200])
print('(u,v,w):')
print(u[43200],v[43200],w[43200])
print('(lat,lon):')
print(lat[43200],lon[43200])
print('delta T')
print(t_range.delta_t[43200])
print('#################################')

### plot ###
fig = plt.figure(figsize=(8,10), facecolor='white')

# altaz plot
ax0 = plt.subplot(2,1,1,projection=ccrs.PlateCarree(central_longitude=0))
ax0.plot(alt.degrees,az.degrees,'b-',transform=ccrs.Geodetic())
ax0.gridlines(draw_labels=True)
#plt.title('ISS orbit on '+str(t_range[0].astimezone(tz).date())+' HKT')

# transit plot
ax1 = plt.subplot(2,1,2,projection=ccrs.Orthographic(central_longitude=158, central_latitude=29))
#ax1.set_extent([113+49/60,114+31/60,22+8/60,22+35/60], crs=ccrs.PlateCarree())
#ax1.set_extent([113,115,22,23], crs=ccrs.PlateCarree())
#ax1.set_extent([115,150,25,60], crs=ccrs.PlateCarree())

shp0 = shapereader.Reader('gadm36_HKG_shp/gadm36_HKG_0.shp')
shp1 = shapereader.Reader('gadm36_HKG_shp/gadm36_HKG_1.shp')
ax1.add_geometries(shp0.geometries(), ccrs.PlateCarree(), facecolor='w',edgecolor='black')
ax1.add_geometries(shp1.geometries(), ccrs.PlateCarree(), facecolor='w',edgecolor='black')
ax1.coastlines()
ax1.set_global()
ax1.gridlines(dms=True,xlocs=[-150,-120,-90,-60,-30,0,30,60,90,120,150,180],xformatter=LONGITUDE_FORMATTER,ylocs=[-60,-30,0,30,60],yformatter=LATITUDE_FORMATTER)
ax1.plot(lon,lat,'b-',markersize=12,transform=ccrs.PlateCarree())
##for i in range(int(len(t_range)/30)):
##    ax1.text(lon[i*30],lat[i*30],t_range[i*30].astimezone(tz).time())

plt.tight_layout()
plt.show()
