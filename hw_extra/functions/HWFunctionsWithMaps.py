import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import SymLogNorm



import numpy as np
import pandas as pd
import seaborn as sns


from scipy.stats import norm
from scipy.optimize import curve_fit


import geopandas as geo
from shapely.geometry import Point


sns.set_theme(rc={'figure.figsize':(20,8.27)})
sns.set_theme(style="darkgrid")
sns.set_palette("pastel")

from HWCompAndStats import get_trends

def KoeppenMap(zone_map, gdf_KM, min_lon, min_lat, max_lon, max_lat,
              path = 'hcarrillo/diagnosis/KoeppenMaps/', filename = 'Koeppen.pdf'):

    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    #sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks")
    sns.set_palette("pastel")

    # Define the color palette used in the PDF
    climate_colors = ["#960000", "#FF0000", "#FF6E6E", "#FFCCCC", "#CC8D14", "#CCAA54", "#FFCC00", "#FFFF64",
                    "#007800", "#005000", "#003200", "#96FF00", "#00D700", "#00AA00", "#BEBE00", "#8C8C00",
                    "#5A5A00", "#550055", "#820082", "#C800C8", "#FF6EFF", "#646464", "#8C8C8C", "#BEBEBE",
                    "#E6E6E6", "#6E28B4", "#B464FA", "#C89BFA", "#C8C8FF", "#6496FF", "#64FFFF", "#F5FFFF"]

    # Create a mapping from numerical values to colors
    unique_values = sorted(gdf_KM['layer'].unique())
    #adjusted_values = [int(val) for val in unique_values]
    color_map = {val: climate_colors[int(val)-1] for val in unique_values}
    print(f"Color map: {color_map}")


    # Plot with custom colors
    fig, ax = plt.subplots(figsize=(5, 5))
    gdf_KM.plot(ax=ax, color=gdf_KM['layer'].map(color_map), legend=True, zorder = 1, edgecolor='none')#, cmap='tab20')  # Adjust cmap if needed

    zone_map.boundary.plot(ax=ax, zorder=2, edgecolor='black', linewidth=0.5)#, alpha=0.1)

    label = ['Af', 'Am', 'As', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb',
             'Cfc', 'Csa', 'Csb', 'Csc', 'Cwa','Cwb', 'Cwc', 'Dfa', 'Dfb', 'Dfc',
             'Dfd', 'Dsa', 'Dsb', 'Dsc', 'Dsd','Dwa', 'Dwb', 'Dwc', 'Dwd', 'EF','ET', 'Ocean']

    # Add legend manually
    handles = [patches.Patch(color=color_map[val], label=label[int(val)-1]) for val in unique_values]#label=f'Class {val}') for val in unique_values]
    ax.legend(handles=handles, title='Climate Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    # limits 
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Customize the plot
    ax.set_xlabel('')
    ax.set_ylabel('')
    #ax.legend(loc = 'center left')
    plt.show()

    fig.savefig(path + filename, format='pdf', bbox_inches='tight')#, dpi=dpi)



def map_stations(stations_df, zonemap, min_lon, min_lat, max_lon, max_lat, stat_id='national_code', fig_PATH = 'notebooks/hcarrillo/', savefile = False, filename = 'Stations.pdf',dpi = 150):
    #Zonemap is something  like zonemap = gpd.read_file('path_to_your_map_file')
    sns.set_theme(rc={'figure.figsize':(10, 4.14)}) #(20,8.27)})
    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")
    
    # Convert the DataFrame to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(stations_df['longitude'], stations_df['latitude'])]
    stations_gdf = geo.GeoDataFrame(stations_df, geometry=geometry)


    # Set the CRS for the stations GeoDataFrame
    stations_gdf.set_crs(epsg=4326, inplace=True)

    # Plot the map
    ax = zonemap.plot(figsize=(5, 5))#10, 10))#, color='white', edgecolor='black')

    ax.set_facecolor("lightgrey")

    # Set the x and y limits to the bounding box
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Plot the stations on the map
    stations_gdf.plot(ax=ax, marker='o', color='red', markersize=100)

    # Add labels for each station
    #for x, y, label in zip(stations_gdf.geometry.x, stations_gdf.geometry.y, stations_gdf.index.tolist()): #stations_gdf[stat_id]):
    for x, y, label in zip(stations_gdf.geometry.x, stations_gdf.geometry.y, range(len(stations_gdf.index.tolist()))):
        ax.text(x, y, str(label+1), fontsize=8, color='black')#ha='right', va='bottom', color='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.savefig(fig_PATH + filename, format='pdf', bbox_inches='tight')#, dpi=dpi)


##########

def KoeppenMapWithStations(stations, zone_map, gdf_KM, min_lon, min_lat, max_lon, max_lat,
              path = 'hcarrillo/diagnosis/KoeppenMaps/', filename = 'KoeppenWithStations.pdf'):

    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    #sns.set_theme(style="darkgrid")
    sns.set_theme(style="ticks")
    sns.set_palette("pastel")

    # Define the color palette used in the PDF
    climate_colors = ["#960000", "#FF0000", "#FF6E6E", "#FFCCCC", "#CC8D14", "#CCAA54", "#FFCC00", "#FFFF64",
                    "#007800", "#005000", "#003200", "#96FF00", "#00D700", "#00AA00", "#BEBE00", "#8C8C00",
                    "#5A5A00", "#550055", "#820082", "#C800C8", "#FF6EFF", "#646464", "#8C8C8C", "#BEBEBE",
                    "#E6E6E6", "#6E28B4", "#B464FA", "#C89BFA", "#C8C8FF", "#6496FF", "#64FFFF", "#F5FFFF"]

    # Create a mapping from numerical values to colors
    unique_values = sorted(gdf_KM['layer'].unique())
    #adjusted_values = [int(val) for val in unique_values]
    color_map = {val: climate_colors[int(val)-1] for val in unique_values}
    print(f"Color map: {color_map}")


    # Plot with custom colors
    fig, ax = plt.subplots(figsize=(5, 5))
    gdf_KM.plot(ax=ax, color=gdf_KM['layer'].map(color_map), legend=True, zorder = 1, edgecolor='none')#, cmap='tab20')  # Adjust cmap if needed

    zone_map.boundary.plot(ax=ax, zorder=2, edgecolor='black', linewidth=0.5)#, alpha=0.1)


    geometry = [Point(xy) for xy in zip(stations['longitude'], stations['latitude'])]
    stations_gdf = geo.GeoDataFrame(stations, geometry=geometry)
    stations_gdf.plot(ax=ax, marker='o', color='red', alpha = 1, zorder = 3, markersize=100)


    # Set the CRS for the stations GeoDataFrame
    stations_gdf.set_crs(epsg=4326, inplace=True)

    label = ['Af', 'Am', 'As', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb',
             'Cfc', 'Csa', 'Csb', 'Csc', 'Cwa','Cwb', 'Cwc', 'Dfa', 'Dfb', 'Dfc',
             'Dfd', 'Dsa', 'Dsb', 'Dsc', 'Dsd','Dwa', 'Dwb', 'Dwc', 'Dwd', 'EF','ET', 'Ocean']

    # Add legend manually
    handles = [patches.Patch(color=color_map[val], label=label[int(val)-1]) for val in unique_values]#label=f'Class {val}') for val in unique_values]
    ax.legend(handles=handles, title='Climate Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    # limits 
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    #ax.set_facecolor("white")


    # Customize the plot
    ax.set_xlabel('')
    ax.set_ylabel('')
    #ax.legend(loc = 'center left')
    plt.show()

    fig.savefig(path + filename, format='pdf', bbox_inches='tight')#, dpi=dpi)


######################


def map_heatwaves_plot(start,end,stations,df,metadata, zone_map):
    #Largo del intervalo
    L=end-start

    #Seleccionar el rango de años
    r1=df[df.index >= start] #[df["Years"]>=start]
    r2=r1[r1.index<end] #r1[r1["Years"]<end]

    #Hacer una lista con el número de olas de calor por estación
    addlist=r2[stations].sum()

    #Crear una lista con los códigos de las estaciones
    codel=[]
    for n in stations:
        codel.append(int(n))

    #Crear listas con las latitudes y longitudes de las estaciones
    latitude = []
    longitude = []
    for st in stations: #i in range(len(stations)):
        latitude.append(float(metadata["latitude"][st])) #.split("/")[1]))
        longitude.append(float(metadata["longitude"][st])) #.split("/")[1]))

    #Crear el geodataframe y graficar
    geo_df=geo.GeoDataFrame(addlist,geometry=geo.points_from_xy(longitude,latitude))
    fig, axs=plt.subplots(1,1)
    geo_df.plot(column=0,ax=axs,cmap='hot_r', markersize=50,zorder=2)#,vmin=4*L,vmax=10*L)#, marker="o",legend=True)#,legend_kwds={'label': "Número de Olas de Calor"})
    map=zone_map.plot(ax=axs,zorder=1)
    map.set_facecolor("lightgrey")
    map.set_title(f"Sum of HWN in {start}-{end}")
    map.set_xlabel("Longitude")
    map.set_ylabel("Latitude")
    #plt.colorbar()
    #map.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))

    plt.show()



def plot_trends_years(start,end,df,stations,zone_map, min_lon, min_lat, max_lon, max_lat, hwdef, yw, hwi, vmin=-1,vmax=1, fig_PATH = 'notebooks/hcarrillo'):
    #Obtener las tendencias y las estaciones utilizables
    trends,usable, p_values = get_trends(start,end,df)

    #Crear listas con las latitudes y longitudes de las estaciones
    coord_list = []

    latitude = stations['latitude']
    longitude = stations['longitude']

    #Crear el geodataframe y graficar
    df = pd.DataFrame(trends,index=[0])
    df = df.transpose()
    geo_df = geo.GeoDataFrame(df,geometry=geo.points_from_xy(longitude,latitude))
    geo1 = geo_df[geo_df[0]<0]
    geo2 = geo_df[geo_df[0]>0]
    geo3 = geo_df[geo_df[0]==0]

    df_sign = pd.DataFrame(p_values,index=[0])
    df_sign = df_sign.transpose()
    geo_df_sign = geo.GeoDataFrame(df_sign,geometry=geo.points_from_xy(longitude,latitude))
    geo1_sign = geo_df_sign[geo_df_sign[0]<0.05]

    sns.set_theme(rc={'figure.figsize':(20,8.27)})
    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    mz = 150

    if geo1_sign.__len__()>0:
        geo1_sign.plot(column=0,ax=ax, color = 'black', linewidth=1, marker='^',zorder=1, markersize=mz+20,legend=True,vmin=vmin,vmax=vmax)#,legend_kwds={'label': "Tendencia en Número de Olas de Calor por Año"})

    if geo1.__len__()>0:
        #geo1.plot(column=0,ax=ax,edgecolor='grey', linewidth=1,cmap='seismic', marker='v',zorder=2, markersize=mz,legend=True,vmin=vmin,vmax=vmax)#,legend_kwds={'label': "Tendencia en Número de Olas de Calor por Año"})
        geo1.plot(column=0,ax=ax,linewidth=1,cmap='seismic', marker='v',zorder=2, markersize=mz,legend=True,vmin=vmin,vmax=vmax)

    else:
        geo2.plot(column=0,ax=ax,linewidth=1,cmap='seismic', marker='^',zorder=1, markersize=mz,legend=True,vmin=vmin,vmax=vmax)#,legend_kwds={'label': "Tendencia en Número de Olas de Calor por Año"})
    
    if geo2.__len__()>0 and geo1.__len__()>0:
        geo2.plot(column=0,ax=ax,linewidth=1,cmap='seismic', marker='^',zorder=1, markersize=mz,legend=False,vmin=vmin,vmax=vmax)
    
    if geo3.__len__()>0:
        geo3.plot(column=0,ax=ax,linewidth=1,cmap='seismic', marker='s',zorder=3, markersize=mz,legend=False,vmin=vmin,vmax=vmax)
    
    map=zone_map.plot(ax=ax,zorder=0,color="darkgrey")
    map.set_facecolor('lightgrey')
    map.set_xlabel("Longitude")
    map.set_ylabel("Latitude")

    # Set the x and y limits to the bounding box
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    #plt.title(f"Tendencias de olas de calor en California de {start} a {end}")
    
    #plt.show()

    fig.savefig(fig_PATH + '/diagnosis/stats_by_hwdef/Trends_' + hwdef + '_' + yw + '_' + hwi + '.pdf', format='pdf', bbox_inches='tight', dpi=150)





def plot_trends_years_all_indices(start,end,df_all_indices,stations,zone_map, min_lon, min_lat, max_lon, max_lat,
                                  hwdef, vmin=-1,vmax=1, fig_PATH = 'notebooks/hcarrillo', zone='Central_Chile',
                                  fsize = [10, 5], wspace = 0.2, format = 'pdf', pos_adj = 1.2):
    
    cmap = 'seismic' #'jet' #'coolwarm' #'seismic'

    linthresh = 0.025
    cmap = plt.cm.seismic #coolwarm
    norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)

    #Crear listas con las latitudes y longitudes de las estaciones
    latitude = stations['latitude']
    longitude = stations['longitude']

    sns.set_theme(style="darkgrid")
    sns.set_palette("pastel")
    fig = plt.figure(figsize=(fsize[0], fsize[1]))

    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.1], wspace=wspace)#0.2)#0)
    ax = [fig.add_subplot(gs[i]) for i in range(6)]

    mz = 70

    trends, r_values, p_values = {}, {}, {}
    for i, hwi in enumerate(list(df_all_indices.keys())):
        #Obtener las tendencias y las estaciones utilizables
        trends[hwi], r_values[hwi], p_values[hwi] = get_trends(start,end,df_all_indices[hwi])


        #Crear el geodataframe y graficar
        df = pd.DataFrame(trends[hwi],index=[0])
        df = df.transpose()
        geo_df = geo.GeoDataFrame(df,geometry=geo.points_from_xy(longitude, latitude))
        geo1 = geo_df[geo_df[0]<0]
        geo2 = geo_df[geo_df[0]>0]
        geo3 = geo_df[geo_df[0]==0]

        df_sign = pd.DataFrame(p_values[hwi],index=[1])
        df_sign = df_sign.transpose()
        geo_df_sign = geo.GeoDataFrame(df_sign,geometry=geo.points_from_xy(longitude,latitude))
        geo1_sign = geo_df_sign[geo_df_sign[1]<0.05]

        df_concat = pd.concat([df, df_sign], axis=1)
        geo_df_concat = geo.GeoDataFrame(df_concat,geometry=geo.points_from_xy(longitude,latitude))
        #print(geo_df_concat)
        geo_up_sign = geo_df_concat[(geo_df_concat[0]>0) & (geo_df_concat[1]<0.05)]
        geo_down_sign = geo_df_concat[(geo_df_concat[0]<0) & (geo_df_concat[1]<0.05)]
        geo_notrend_sign = geo_df_concat[(geo_df_concat[0]==0) & (geo_df_concat[1]<0.05)]
        geo_up_nosign = geo_df_concat[(geo_df_concat[0]>0) & (geo_df_concat[1]>=0.05)]
        geo_down_nosign = geo_df_concat[(geo_df_concat[0]<0) & (geo_df_concat[1]>=0.05)]
        geo_notrend_nosign = geo_df_concat[(geo_df_concat[0]==0) & (geo_df_concat[1]>=0.05)]
        print(hwi)
        #print(geo_up_sign)

        if geo_up_sign.__len__()>0:
            geo_up_sign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap, norm=norm, linewidth=1, marker='^',zorder=2, markersize=mz+50,legend=False,vmin=vmin,vmax=vmax)
        if geo_up_nosign.__len__()>0:
            geo_up_nosign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap, norm=norm,linewidth=1, marker='^',zorder=1, markersize=mz-20,legend=False,vmin=vmin,vmax=vmax)
        if geo_down_sign.__len__()>0:
            geo_down_sign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap,norm=norm, linewidth=1, marker='^',zorder=2, markersize=mz+50,legend=False,vmin=vmin,vmax=vmax)
        if geo_down_nosign.__len__()>0:
            geo_down_nosign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap, norm=norm,linewidth=1, marker='v',zorder=1, markersize=mz-20,legend=False,vmin=vmin,vmax=vmax)
        if geo_notrend_sign.__len__()>0:
            geo_notrend_sign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap, norm=norm,linewidth=1, marker='o',zorder=2, markersize=mz+50,legend=False,vmin=vmin,vmax=vmax)
        if geo_notrend_nosign.__len__()>0:
            geo_notrend_nosign.plot(column=0,ax=ax[i], edgecolor = 'grey', cmap=cmap, norm=norm,linewidth=1, marker='o',zorder=1, markersize=mz-20,legend=False,vmin=vmin,vmax=vmax)
    
        map=zone_map.plot(ax=ax[i],zorder=0,color="darkgrey")
        map.set_facecolor('lightgrey')
        map.set_xlabel("Longitude")
        if i == 0:
            map.set_ylabel("Latitude")
        else:
            ax[i].set_yticklabels([])

        # Set the x and y limits to the bounding box
        ax[i].set_xlim(min_lon, max_lon)
        ax[i].set_ylim(min_lat, max_lat)

        ax[i].set_title(hwi)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # Dummy mappable object to be used for colorbar

    # Add the colorbar to the last subplot area
    ax[-1].set_yticklabels([])
    ax[-1].set_xticklabels('')
    ax[-1].set_title('')
    cax = fig.add_subplot(gs[-1])
    cbar = fig.colorbar(sm, cax=cax)#, label='Slope')
    cbar.set_label('Slope', rotation=-90, labelpad=10)  # Rotate the label 270 degrees
    
    # Manually adjust the positions of the subplots to reduce space
    for i in range(5):
        pos1 = ax[i].get_position() # get the original position
        #pos2 = [pos1.x0, pos1.y0, pos1.width * 1.2, pos1.height]
        pos2 = [pos1.x0, pos1.y0, pos1.width * pos_adj, pos1.height]
        ax[i].set_position(pos2) # set a new position

    fig.savefig(fig_PATH + 'Trends_' + zone +'_Indices_' + hwdef + '.' + format, format=format, bbox_inches='tight', dpi=150)
    return trends, r_values, p_values

