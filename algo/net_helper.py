import geopandas as gpd
import pandas as pd
import os

cols_static = ["edge_id", "osm_id", "length", "geometry"]
cols_dir = ["access_bicycle", "access_pedestrian", "index_bike", "index_walk"]

def netascore_generate_subset(dir, in_file, out_file, clip_geom):
    tmp = gpd.read_file(os.path.join(dir, in_file), layer="edge", rows=10)
    srid = tmp.crs
    cg = clip_geom.to_crs(srid).envelope
    print(cg)
    edges = gpd.read_file(os.path.join(dir, in_file), layer="edge", mask=cg)
    edges.to_file(os.path.join(dir, out_file), layer="edge", driver="GPKG")
    # temp workaround: add "node_id" column to maintain original node id after subsetting
    nodes = gpd.read_file(os.path.join(dir, in_file), layer="node")
    nodes["node_id"] = nodes.index + 1 # add node_id column
    nodes.to_file(os.path.join(dir, in_file + "_nid"), layer="node", driver="GPKG")
    nodes = gpd.read_file(os.path.join(dir, in_file + "_nid"), layer="node", mask=cg)
    nodes.to_file(os.path.join(dir, out_file), layer="node", driver="GPKG")
    return cg

def netascore_to_routable_net(netascore_gdf: gpd.GeoDataFrame):
    # filter input df
    cols = cols_static + ["from_node", "to_node"] + [f"{x}_ft" for x in cols_dir] + [f"{x}_tf" for x in cols_dir]
    net_a = netascore_gdf.filter(cols, axis=1).copy()
    # append inverted state
    net_a["inv"] = False
    # generate mapping for renaming dir-specific columns
    mapping = {f'{k}_ft': f'{k}_tf' for k in cols_dir}
    mapping.update({f'{k}_tf': f'{k}_ft' for k in cols_dir})
    mapping.update({"from_node":"to_node", "to_node":"from_node"})
    net_b = net_a.rename(columns=mapping)
    net_b["inv"] = True
    # append inverted net
    net = net_a.append(net_b, ignore_index=True)
    # remove inverted-dir columns
    net.drop([f'{k}_tf' for k in cols_dir], axis=1, inplace=True)
    return net

def add_centr_to_netascore(netascore_gdf: gpd.GeoDataFrame, centrality, centr_name: str, edge_key: str):
    # convert from dict with compound key to pandas df
    tdf = pd.DataFrame(list(centrality.keys()))
    tdf.rename(columns={0:"from_node", 1:"to_node", 2:"edge_key"}, inplace=True)
    tdf["centrality"] = centrality.values()
    # map centrality value back to original (geo)pandas df (for both directions)
    net_tmp = netascore_gdf.merge(tdf, left_on=["from_node", "to_node", edge_key], right_on=["from_node", "to_node", "edge_key"], how="left", suffixes=[None, "_b"])
    net_tmp.rename(columns={"centrality":"centrality_ft"}, inplace=True)
    net_ready = net_tmp.merge(tdf, left_on=["to_node", "from_node", edge_key], right_on=["from_node", "to_node", "edge_key"], how="left", suffixes=[None, "_c"])
    net_ready.rename(columns={"centrality":"centrality_tf"}, inplace=True)
    net_ready.set_index(edge_key, inplace=True)
    
    netascore_gdf[f"centr_{centr_name}_ft"] = net_ready.centrality_ft
    netascore_gdf[f"centr_{centr_name}_tf"] = net_ready.centrality_tf
    netascore_gdf[f"centr_{centr_name}_sum"] = net_ready.centrality_tf + net_ready.centrality_ft