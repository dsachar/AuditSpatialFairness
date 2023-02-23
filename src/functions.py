from rtree import index
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import math
import folium


#### The dataframe should have columns lat, lon, label
def load_data(filename):
    df = pd.read_csv(filename, index_col=0)
    df.reset_index(drop=True, inplace=True)

    return df


def get_stats(df, label):
    N = len(df)
    P = df.loc[df[label] == 1, label].count()

    return N, P


def create_rtree(df):
    rtree = index.Index()

    for idx, row in df.iterrows():
        left, bottom, right, top = row['lon'], row['lat'], row['lon'], row['lat']
        rtree.insert(idx, (left, bottom, right, top))
    
    return rtree


def filterbbox(df, min_lon, min_lat, max_lon, max_lat):
    df = df.loc[df['lon'] >= min_lon]
    df = df.loc[df['lon'] <= max_lon]
    df = df.loc[df['lat'] >= min_lat]
    df = df.loc[df['lat'] <= max_lat]
    df.reset_index(drop=True, inplace=True)    
    
    return df


def get_true_types(df, label):
    array = np.array(df[label].values.tolist())
    array[array==3] = 0 ## replace entries with label 3 to have label 0 (for the LAR dataset)
    return array


def get_random_types(N, P):
    return np.random.binomial(size=N, n=1, p=P/N)



def get_simple_stats(points, types):
    n = len(points)
    p = types[points].sum()
    if n>0:
        rho = p/n
    else:
        rho = np.nan

    return (n, p, rho)


def compute_pos_rate(points, types):
    n = len(points)
    p = types[points].sum()
    return p/n


# get the coords of a point id
def id2loc(df, point_id):
    lat = df.loc[[point_id]]['lat'].values[0]
    lon = df.loc[[point_id]]['lon'].values[0]
    return (lat, lon)



def query_range_box(df, rtree, xmin, xmax, ymin, ymax):
    
    
    left, bottom, right, top = xmin, ymin, xmax, ymax

    result = list( rtree.intersection((left, bottom, right, top)) )

    return result


def query_range(df, rtree, center, radius):
    ## for now returns points within square
    
    lat, lon = id2loc(df, center)

    left, bottom, right, top = lon - radius, lat - radius, lon + radius, lat + radius
    result = list( rtree.intersection((left, bottom, right, top)) )

    # tmp_result = []
    # for point in result:
    #     p_lat, p_lon = id2loc(df, point)
    #     dist = math.sqrt( (p_lon-lon)**2 + (p_lat-lat)**2 )
    #     if dist <= radius:
    #         tmp_result.append(point)
    # result = tmp_result


    return result


def query_nn(df, rtree, center, k):
    lat, lon = id2loc(df, center)

    return list(rtree.nearest( [lon, lat], k))


def create_seeds(df, rtree, n_seeds):
    
    # Compute clusters with k-means
    X = df[['lon', 'lat']].to_numpy()
    kmeans = KMeans(n_clusters=n_seeds, n_init='auto').fit(X)
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    
    # Pick seeds from cluster centroids
    seeds = []
    for c in cluster_centers:
        seeds.append(list(rtree.nearest([c[0], c[1]], 1))[0])
    
    return seeds


def compute_max_likeli(n, p, N, P):
    
    ## l1max =  p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)
    ## handle extreme cases

    rho = P/N
    l0max = P*math.log(rho) + (N-P)*math.log(1-rho)

    if n == 0 or n == N: ## rho_in == 0/0 or rho_out == 0/0
        l1max = l0max
        return l1max


    rho_in = p/n
    rho_out = (P-p)/(N-n)


    if p == 0: ## rho_in == 0
        l1max = P*math.log(rho_out) + (N-n - P)*math.log(1-rho_out)
    elif p == n and p == P: ## rho_in == 1 and rho_out == 0
        l1max = 0
    elif p == n: ## rho_in == 1
        l1max = (P-p)*math.log(rho_out) + (N-P)*math.log(1-rho_out)
    elif p == P: ## rho_out == 0
        l1max = p*math.log(rho_in) + (n-p)*math.log(1-rho_in)
    else:
        l1max =  p*math.log(rho_in) + (n-p)*math.log(1-rho_in) + (P-p)*math.log(rho_out) + (N-n - (P-p))*math.log(1-rho_out)

    return l1max



def compute_statistic(n, p, N, P, direction='both', verbose=False):
    ## l1max - l0max

    if verbose:
        print(f'{n=}, {p=}')


    if n == 0 or n == N: ## rho_in == 0/0 or rho_out == 0/0
        return 0 


    rho = P/N
    rho_in = p/n
    rho_out = (P-p)/(N-n)

    if verbose:
        print(f'{rho=}, {rho_in=}, {rho_out=}')
    
    l0max = P*math.log(rho) + (N-P)*math.log(1-rho)    


    if direction == 'less_in':
        ### inside < outside
        if rho_in < rho_out:
            l1max = compute_max_likeli(n, p, N, P)
        else:
            l1max = l0max
    elif direction == 'less_out':
       ### inside > outside
        if rho_in > rho_out:
            l1max = compute_max_likeli(n, p, N, P)
        else:
            l1max = l0max
    else:
        ### inside != outside
        l1max = compute_max_likeli(n, p, N, P)

    statistic = l1max - l0max

    if verbose:
        print(f'{l0max=}, {l1max=}, {statistic=}')

    return statistic 




def create_regions(df, rtree, seeds, radii):
    regions = []
    for seed in seeds:
        for radius in radii: 
            points = query_range(df, rtree, seed, radius)
            region = {
                'points' : points,
                'center' : seed,
                'radius' : radius,  
            }
            regions.append(region)
    
    return regions


def scan_regions(regions, types, N, P, direction='both', verbose=False):
    """ computes the statistic for all regions, and returns the region with max likelihood and the max likelihood """
    statistics = []

    for region in regions:
        n, p, rho = get_simple_stats(region['points'], types)

        statistics.append(compute_statistic(n, p, N, P, direction=direction))
    
    idx = np.argmax(statistics)

    max_likelihood = statistics[idx]

    if verbose:
        print('range', np.amin(statistics), np.amax(statistics))
        print('max likelihood', max_likelihood)
        n, p, rho = get_simple_stats(regions[idx]['points'], types)
        # print(f"at ({regions[idx]['center']}, {regions[idx]['radius']})" )
        compute_statistic(n, p, N, P, direction=direction, verbose=verbose)
    
    return regions[idx], max_likelihood, statistics



def scan_alt_worlds(n_alt_worlds, regions, N, P, verbose=False):
    """ returns all alt worlds sorted by max likelihood, and the max likelihood """
    alt_worlds = []
    for _ in range(n_alt_worlds):
        alt_types = get_random_types(N, P)
        alt_best_region, alt_max_likeli, _ = scan_regions(regions, alt_types, N, P, verbose=verbose)
        alt_worlds.append((alt_types, alt_best_region, alt_max_likeli))

    alt_worlds.sort(key=lambda x: -x[2])

    return alt_worlds, alt_worlds[0][2]


def get_signif_threshold(signif_level, n_alt_worlds, regions, N, P):
    """ returns a statistic value such any region with statistic above that value is unfair at significance level `signif_level`; i.e., has p-value lower than `signif_level`  """
    alt_worlds, _ = scan_alt_worlds(n_alt_worlds, regions, N, P)

    k = int(signif_level * n_alt_worlds)

    signif_thresh = alt_worlds[k][2] ## get the max likelihood at position k

    return signif_thresh



######## partioning-based scan


def scan_partitioning(regions, types):
    rhos = []
    for region in regions:
        n = len(region['points'])
        p = types[region['points']].sum()
        if n>0:
            rho = p/n
        else:
            rho = np.nan
        rhos.append(rho)
    mean_rho = np.nanmean(rhos)
    rhos = np.array(rhos)
    # print('mean_rho', mean_rho)
    # print(rhos[:10])
    scores = (rhos - mean_rho)**2

    max_score = np.nanmax(scores)
    idx = np.nanargmax(scores)

    # print(scores[:10])
    # print(np.nanmax(scores), np.nanargmax(scores) )
    return regions[idx], max_score, scores


######## create synthetic datasets

def create_points(n, rho):
    points = []
    types = np.random.binomial(size=n, n=1, p=rho)
    
    # guarantee n * rho positives
    n_pos = int(n * rho)
    while np.sum(types) != n_pos:
        idx = np.random.randint(0,n)
        if np.sum(types) > n_pos and types[idx]==1:
            types[idx] = 0
        elif np.sum(types) < n_pos and types[idx]==0:
            types[idx] = 1

    for i in range(n):
        x = random.random()
        y = random.random()
        points.append((x,y))
        
    return points, types




######## draw map functions

def show_grid_region(df, grid_info, types, region):

    lon_min = grid_info['lon_min']
    lon_max = grid_info['lon_max']
    lat_min = grid_info['lat_min']
    lat_max = grid_info['lat_max']

    lon_n = grid_info['lon_n']
    lat_n = grid_info['lat_n']


    i, j = region['grid_loc']

    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, tiles="Stamen Toner")

    # pos_group = folium.FeatureGroup("Positive")
    # neg_group = folium.FeatureGroup("Negative")

    lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
    lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

    lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
    lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

    n, p, rho = get_simple_stats(region['points'], types)

    folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)], tooltip=f'{n=}, {p=}, rho={rho:.2f}').add_to( mapit )


    for point in region['points']:
        if types[point] == 1:
            # pos_group.add_child(folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ))
            folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        else:
            # neg_group.add_child(folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ))
            folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )

    # mapit.add_child(pos_group)
    # mapit.add_child(neg_group)
    mapit.fit_bounds([ [lat_start, lon_start],[lat_end, lon_end] ])
    return mapit



def show_grid_regions(df, grid_info, types, regions):

    lon_min = grid_info['lon_min']
    lon_max = grid_info['lon_max']
    lat_min = grid_info['lat_min']
    lat_max = grid_info['lat_max']

    lon_n = grid_info['lon_n']
    lat_n = grid_info['lat_n']


    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, tiles="Stamen Toner")

    for region in regions:

        i, j = region['grid_loc']


        lon_start = lon_min + (i/lon_n)*(lon_max - lon_min)
        lon_end = lon_min + ((i+1)/lon_n)*(lon_max - lon_min)

        lat_start = lat_min + (j/lat_n)*(lat_max - lat_min)
        lat_end = lat_min + ((j+1)/lat_n)*(lat_max - lat_min)

        n, p, rho = get_simple_stats(region['points'], types)

        folium.Rectangle([(lat_start,lon_start), (lat_end,lon_end)], tooltip=f'{n=}, {p=}, ρ={rho:.2f}').add_to( mapit )

        for point in region['points']:
            if types[point] == 1:
                folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
            else:
                folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )

    mapit.fit_bounds([(lat_min, lon_min), (lat_max, lon_max)])
    return mapit



def show_circular_region(df, types, region):

    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, tiles="Stamen Toner")

    r = region['radius'] * 111320 ## roughly convert diff in lat/lon to meters

    n, p, rho = get_simple_stats(region['points'], types)

    folium.Circle(location=id2loc(df, region['center']), color='#0000FF', fill_color='#0000FF', fill=True, opacity=0.4, fill_opacity=0.4, radius=r, tooltip=f'{n=}, {p=}, rho={rho:.2f}' ).add_to( mapit )
    
    for point in region['points']:
        if types[point] == 1:
            folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        else:
            folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
    
    return mapit


def show_circular_regions(df, types, regions):
    
    mapit = folium.Map(location=[37.09, -95.71], zoom_start=5, tiles="Stamen Toner")

    for region in regions:
        n, p, rho = get_simple_stats(region['points'], types)
        
        # r = region['radius'] * 111320 ## roughly convert diff in lat/lon to meters
        # folium.Circle(location=id2loc(df, region['center']), color='#0000FF', fill_color='#0000FF', fill=True, opacity=0.4, fill_opacity=0.4, radius=r, tooltip=f'{n=}, {p=}, rho={rho:.2f}' ).add_to( mapit )
        
        r = region['radius']
        c = id2loc(df, region['center'])
        folium.Rectangle([(c[0]-r, c[1]-r), (c[0]+r, c[1]+r)], tooltip=f'{n=}, {p=}, ρ={rho:.2f}').add_to( mapit )

        for point in region['points']:
            if types[point] == 1:
                folium.CircleMarker( location=id2loc(df, point), color='#00FF00', fill_color='#00FF00', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
            else:
                folium.CircleMarker( location=id2loc(df, point), color='#FF0000', fill_color='#FF0000', fill=True, opacity=0.4, fill_opacity=0.4, radius=2 ).add_to( mapit )
        
    return mapit

