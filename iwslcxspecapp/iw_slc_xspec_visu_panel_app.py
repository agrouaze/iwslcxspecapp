"""
A Grouazel
SARWAVE L1B IW SLC product exploration (visu xspectra inter and intra)
usage:
conda activate py3_panel
 panel serve iw_slc_xspec_visu_panel_app.py --allow-websocket-origin=hostname:5006
"""
import panel as pn
import numpy as np
import holoviews as hv
import datatree
import xarray as xr
import logging
import copy
import hvplot.pandas
import hvplot.xarray  # noqa
import cartopy.crs as ccrs
import geoviews as gv
import xsar
import sys
import get_path_from_base_SAFE
import match_GRD_SLC
from symmetrize_xspec import symmetrize_xspectrum
import os
import glob
from holoviews.operation.datashader import datashade, rasterize
from matplotlib import colors as mcolors
import pandas as pd
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.plotting import figure
import yaml
# Read YAML file
with open("data.yaml", 'r') as stream:
    data_loaded = yaml.safe_load(stream)
# L1B_file_default = "/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20180104T061957_20180104T062025_020001_02211F_77E6.SAFE_L1B_xspec_IFR_VV_0.1.nc"
L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20210413T043041_20210413T043109_037427_04695F_FE5C.SAFE_L1B_xspec_IFR_0.3.nc'
L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20220921T063225_20220921T063252_045099_05639C_D284.SAFE/s1a-iw2-slc-vv-20220921t063225-20220921t063251-045099-05639c-005_L1B_xspec_IFR_0.6.nc'
L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20220921T063225_20220921T063252_045099_05639C_D284.SAFE/s1a-iw2-slc-vv-20220921t063225-20220921t063251-045099-05639c-005_L1B_xspec_IFR_0.6.nc'
L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20170907T103019_20170907T103047_018268_01EB76_5F55.SAFE/s1a-iw2-slc-vv-20170907t103020-20170907t103045-018268-01eb76-005_L1B_xspec_IFR_0.6.nc'
# subswath = 'subswath_1'
#subswath = 'iw1_vv'
L1B_file_default = data_loaded['datadir']
main_map_width = 600
main_map_height = 500


################################################
def get_tile_corner_image_idx(ds, bursti, tile_line_i, tile_sample_i):
    """

    Parameters
    ----------
    ds
    bursti
    tile_line_i
    tile_sample_i

    Returns
    -------

    """
    top_line = ds['line'].isel({"burst": bursti, "tile_line": tile_line_i}).values + ds.tile_nperseg_line / 2
    print('ds[sample]', ds['sample'])
    left_sample = ds['sample'].isel({"burst": bursti, "tile_sample": tile_sample_i}).values - ds.tile_nperseg_sample / 2
    # left_sample = ds['sample'].isel({"tile_sample": tile_sample_i}).values - ds.tile_nperseg_sample / 2
    bottom_line = ds['line'].isel({"burst": bursti, "tile_line": tile_line_i}).values - ds.tile_nperseg_line / 2
    # bottom_line = ds['line'].isel({"tile_line": tile_line_i}).values - ds.tile_nperseg_line / 2
    right_sample = ds['sample'].isel(
        {"burst": bursti, "tile_sample": tile_sample_i}).values + ds.tile_nperseg_sample / 2
    # right_sample = ds['sample'].isel(
    #    {"tile_sample": tile_sample_i}).values + ds.tile_nperseg_sample / 2

    lines = [top_line, top_line, bottom_line, bottom_line, top_line]
    samples = [left_sample, right_sample, right_sample, left_sample, left_sample]
    return lines, samples


def read_data_L1B(L1B_file, typee='intra'):
    # parentdir = os.path.basename(os.path.dirname(L1B_file))
    dt = datatree.open_datatree(L1B_file)
    if typee == 'intra':
        # ds = dt[subswath]['intraburst_xspectra'].to_dataset()
        ds = dt['intraburst_xspectra'].to_dataset()
    else:
        # ds = dt[subswath]['interburst_xspectra'].to_dataset()
        ds = dt['interburst_xspectra'].to_dataset()
    return ds


#
def get_tabulated_intra_burst_data(ds):
    """

    Parameters
    ----------
    ds: xarr.Dataset L1B

    Returns
    -------

    """
    allons = []
    allats = []
    altileline = []
    altilesample = []
    alburst = []
    for iburst in range(ds.burst.size):
        for itilesample in range(ds['corner_longitude'].shape[1]):
            for itileline in range(ds['corner_longitude'].shape[3]):
                allons.append(
                    ds['longitude'].isel({'burst': iburst, 'tile_line': itileline, 'tile_sample': itilesample}).values)
                allats.append(
                    ds['latitude'].isel({'burst': iburst, 'tile_line': itileline, 'tile_sample': itilesample}).values)
                altileline.append(itileline)
                altilesample.append(itilesample)
                alburst.append(iburst)
    # cds = ColumnDataSource(pd.DataFrame({'longitude':allons,'latitude':allats,'Tsample':altilesample,
    #                                      'Tline':altileline,'burst':alburst}))
    # cds = hv.Dataset(data={'longitude':allons,'latitude':allats,'Tsample':altilesample,'Tline':altileline,'burst':alburst})
    cds = pd.DataFrame(
        {'longitude': allons, 'latitude': allats, 'Tsample': altilesample, 'Tline': altileline, 'burst': alburst})
    return cds


#
#

#

#
#
def add_cartesian_wavelength_circles(default_values=[100, 300, 600]):
    """

    Parameters
    ----------
    default_values list of ntegers

    Returns
    -------

    """

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return (x, y)

    N_pts = 100
    phi = np.linspace(0, 2 * np.pi, N_pts)
    res = {}
    for rru in default_values:
        r100 = np.ones(N_pts) * 2 * np.pi / rru
        coords_cart_100 = []
        for uu in range(N_pts):
            coords_cart_100.append(pol2cart(r100[uu], phi[uu]))
        coords_cart_100 = np.array(coords_cart_100)
        res[rru] = coords_cart_100
    all_circles = []
    for cck in res:
        ccv = res[cck]
        ccim = hv.Curve((ccv[:, 0], ccv[:, 1]), label='%s m' % cck)
        all_circles.append(ccim)
    return hv.Overlay(all_circles)


#
#
def display_xspec_cart_holo(ds, bu=0, li=0, sam=0, typee='Re'):
    """

    Parameters
    ----------
    ds: dataset xarray with xspectra L1B IW intra or inter burst for one subswath
    bu: int
    li: int
    sam: int
    typee :str

    Returns
    -------

    """

    if typee == 'Re':
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "violet", "mediumpurple", "cyan", "springgreen",
                                                              "yellow", "red"])
    else:
        cmap = 'PuOr'

    set_xspec = ds[{'burst': bu, 'tile_line': li, 'tile_sample': sam}]
    # if False:
    #     set_xspec = set_xspec.assign_coords({'ky': set_xspec['k_az'], 'kx': set_xspec['k_rg']})
    #     set_xspec = set_xspec.swap_dims({'freq_line': 'ky', 'freq_sample': 'kx'})
    #     kx_varname = 'kx'
    #     ky_varname = 'ky'
    # else:
    set_xspec = set_xspec.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
    set_xspec = symmetrize_xspectrum(set_xspec, dim_range='k_rg', dim_azimuth='k_az')
    kx_varname = 'k_rg'
    ky_varname = 'k_az'
    sp = set_xspec['xspectra_2tau_%s' % typee].mean(dim='2tau')
    sp2 = sp.where(np.logical_and(np.abs(sp[kx_varname]) <= 0.14, np.abs(sp[ky_varname]) <= 0.14), drop=True)

    hv.extension('bokeh')
    if typee == 'Re':
        im = hv.Image(abs(sp2), kdims=[kx_varname, ky_varname]).opts(width=small_plot_with, height=small_plot_height,
                                                                     cmap=cmap, colorbar=True, xlim=(
                -0.07, 0.07))  # gist_ncar_r , 'cet_linear_wyor_100_45_c55'
    else:
        if np.all(np.isnan(sp2)):
            extrema = 1
        else:
            extrema = float(abs(sp2).max().values)

        im = hv.Image(sp2, kdims=[kx_varname, ky_varname]).opts(width=small_plot_with, height=small_plot_height,
                                                                cmap=cmap, colorbar=True, xlim=(-0.07, 0.07),
                                                                show_grid=True, clim=(-extrema, extrema))
    tt = hv.Text(-0.06, 0.05, 'burst %s\ntile range: %s\ntile azimuth: %s' % (bu, sam, li), halign='left', fontsize=8)
    cc = add_cartesian_wavelength_circles(default_values=[100, 300, 600])
    return im * tt * cc


#
#

class monAppIW_SLC:
    def __init__(self):
        self.previous_xspec_selected = 0
        self.latest_click = 0
        # prepare data
        logging.info('ok init')
        self.ds_intra = None
        self.cds_intra = None
        self.ds_inter = None
        self.cds_inter = None
        self.l1bpath = None
        self.subswath = None
        self.xsarobjslc = None
        self.xsarobjgrd = None
        self.burst_type = 'intra'

    def display_intra_inter_burst_grids(self):
        """

        Parameters
        ----------
        ds dataset xarray one sub-swath intra or inter burst
        cds pandas DataFrame computed from ds
        burst_type str intra or inter

        Returns
        -------

        """
        all_poly = []
        if self.burst_type == 'intra':
            ds = self.ds_intra
            cds = self.cds_intra
        else:
            ds = self.ds_inter
            cds = self.cds_inter
        ds['corner_longitude'] = ds['corner_longitude'].persist()
        ds['corner_latitude'] = ds['corner_latitude'].persist()
        for iburst in range(ds.burst.size):
            for itilesample in range(ds['corner_longitude'].shape[1]):
                for itileline in range(ds['corner_longitude'].shape[3]):
                    clon = ds['corner_longitude'].isel(
                        {'burst': iburst, 'tile_line': itileline, 'tile_sample': itilesample}).values.ravel(order='A')
                    clat = ds['corner_latitude'].isel(
                        {'burst': iburst, 'tile_line': itileline, 'tile_sample': itilesample}).values.ravel(order='A')
                    clon2 = copy.copy(clon)
                    clat2 = copy.copy(clat)
                    clat2[2] = clat[3]
                    clat2[3] = clat[2]
                    clon2[2] = clon[3]
                    clon2[3] = clon[2]
                    tmpo = np.stack([clon2, clat2]).T
                    tmpo = np.vstack([tmpo, tmpo[0, :]])
                    tmppoly = gv.Path((tmpo[:, 0], tmpo[:, 1]), kdims=['Longitude', 'Latitude'])
                    all_poly.append(tmppoly)
        projection = ccrs.PlateCarree()
        if self.burst_type == 'intra':
            coco = 'blue'
        else:
            coco = 'red'
        points = cds.hvplot.points(x='longitude', y='latitude', hover_cols='all', use_index=False,
                                   projection=projection, label=self.burst_type, color=coco).opts(tools=['hover'],
                                                                                                  size=10)
        # points = gv.Points(('longitude','latitude'),source=cds).opts(tools=['hover'])
        # gv.tile_sources.Wikipedia  *
        res = (gv.tile_sources.Wikipedia * gv.Overlay(all_poly) * points).opts(width=main_map_width,
                                                                               height=main_map_height,
                                                                               show_legend=True, title=os.path.basename(
                self.l1bpath) + '\n' + self.subswath,
                                                                               fontsize={'title': 8})
        return res

    def display_roughness_slc(self, l1b_path, subswath, burst, tile_sample_id, tile_line_id, dsl1b):
        """

        Parameters
        ----------
        l1b_path: str
        subswath: str subswath_2 for example
        burst: int
        tile_sample_id: int
        tile_line_id: int
        dsl1b: xarray.Dataset  L1B

        Returns
        -------

        """
        print('start roughness display')
        lines, samples = get_tile_corner_image_idx(dsl1b, bursti=burst,
                                                   tile_line_i=tile_line_id,
                                                   tile_sample_i=tile_sample_id)
        print('lines', lines)
        lines = np.array(lines).astype(int)
        samples = np.array(samples).astype(int)
        start_line = lines.min()
        stop_line = lines.max()
        start_sample = samples.min()
        stop_sample = samples.max()
        rough = abs(self.xsarobjslc.dataset['digital_number'].isel(
            {"line": slice(start_line, stop_line), 'sample': slice(start_sample, stop_sample), 'pol': 0}))
        p95percentile = np.percentile(rough.values.squeeze(), 95)
        return rasterize(hv.Image(rough, kdims=['sample', 'line']).opts(cmap='gray', colorbar=True, tools=['hover'],
                                                                        title="%s" % rough.pol.values, width=400,
                                                                        height=300, clim=(0, p95percentile),
                                                                        shared_axes=False))  #

    def display_roughness_grd(self):
        """
        return the nice display from GRD product
        Parameters
        ----------
        l1b_path: str of SLC product
        burst: int
        tile_sample_id: int
        tile_line_id: int
        dsl1b: xarray.Dataset  L1B

        Returns
        -------

        """
        rough = self.xsarobjgrd.dataset['sigma0'].rio.reproject('epsg:4326', shape=(1000, 1000), nodata=np.nan).isel(
            {'pol': 0})
        print(rough)
        # rough = abs(self.xsarobjgrd.dataset['sigma0'].isel({ 'pol': 0})).values.ravel()

        p95percentile = np.nanpercentile(rough.squeeze().values.ravel(), 95)
        print('p95percentile GRD', p95percentile)
        # lons = self.xsarobjgrd.dataset['longitude'].values.ravel()
        # lats = self.xsarobjgrd.dataset['latitude'].values.ravel()
        # print('lats,',lats.shape)
        # data3d = np.stack([lons,lats,rough]).T
        # print('data3d;shape',data3d.shape)
        # res = rasterize(hv.Scatter(data3d,kdims=['longitude', 'latitude']
        #                             ).opts(cmap='gray', colorbar=True, tools=['hover'],
        #                             title="%s" % self.xsarobjgrd.dataset['sigma0'].pol.values, width=400,
        #                             height=300, clim=(0, p95percentile),shared_axes=False,alpha=4,color='z'),
        #                  )  #kdims=['longitude', 'latitude']  vdims=['longitude', 'latitude', 'z']
        # res = hv.Image(rough).opts(cmap='gray', colorbar=False, tools=[],
        #                             title="%s" % rough.pol.values, width=main_map_width,
        #                             height=main_map_height,shared_axes=False,alpha=0.4,clim=(0, p95percentile))*gv.Path(self.xsarobjgrd.footprint)

        # rough2 = rough.assign_coords({'longitude':'x','latitude':'y'})
        rough2 = rough.assign_coords({
            'Longitude': xr.DataArray(rough['x'].values, dims=['x']),
            'Latitude': xr.DataArray(rough['y'].values, dims=['y']),
        }
        )
        rough2 = rough2.swap_dims({'x': 'Longitude', 'y': 'Latitude'})
        new_ds = xr.Dataset()
        new_ds['s0grd'] = rough2
        # res = gv.Image(rough2,label='grd').opts(cmap='gray', colorbar=False, tools=[],
        #                             title="%s" % rough.pol.values, width=main_map_width,show_legend=True,
        #                             height=main_map_height,shared_axes=True,alpha=0.4,clim=(0, p95percentile))*gv.Path(self.xsarobjgrd.footprint)
        # res = new_ds.hvplot.image(x='Longitude',y='Latitude',z="s0grd") # OK but not more tap tool
        res = gv.Path(self.xsarobjgrd.footprint)
        return res

    def update_app_burst(self, burst_type, subswath_id, L1B_file=L1B_file_default):
        """

        Parameters
        ----------
        burst_type:  str
        subswath_id: str
        L1B_file: str one subswath .nc file (previous version it was one nc file for a SAFE)

        Returns
        -------

        """
        # if L1B_file==[]:
        #     L1B_file = L1B_file_default #security util???
        if isinstance(L1B_file, list):
            if len(L1B_file) != 0:
                L1B_file = L1B_file[0]
                print('new L1B ', L1B_file)
            else:
                L1B_file = L1B_file_default  # security util???
        # prepare data
        logging.info('ouai')
        print('L1B_file', L1B_file)
        if L1B_file != self.l1bpath or subswath_id != self.subswath or burst_type != self.burst_type:
            self.l1bpath = L1B_file
            subswath_id = os.path.basename(L1B_file).split('-')[1]+'_'+os.path.basename(L1B_file).split('-')[3]
            self.subswath = subswath_id
            self.burst_type = burst_type
            self.ds_intra = read_data_L1B(L1B_file, typee='intra')
            self.cds_intra = get_tabulated_intra_burst_data(self.ds_intra)
            self.ds_inter = read_data_L1B(L1B_file, typee='intra')
            self.cds_inter = get_tabulated_intra_burst_data(self.ds_inter)

            # base = os.path.basename(self.l1bpath).split('_L1B')[0]
            base = os.path.basename(os.path.dirname(self.l1bpath))
            print('base', base)
            fullpath_safeL1SLC = get_path_from_base_SAFE.get_path_from_base_SAFE(base)
            print('fullpath_safeL1SLC', fullpath_safeL1SLC)
            if subswath_id is not None:
                subswath_nb = subswath_id.split('_')[0][-1]
            str_gdal = 'SENTINEL1_DS:%s:IW%s' % (fullpath_safeL1SLC, subswath_nb)
            print('str_gdal', str_gdal)
            self.xsarobjslc = xsar.Sentinel1Dataset(str_gdal)  # ,resolution='10m'
            grdh_path = match_GRD_SLC.match_SLC_GRD(os.path.basename(fullpath_safeL1SLC), type_seek='GRDH')
            if grdh_path:
                self.xsarobjgrd = xsar.Sentinel1Dataset(grdh_path, resolution='200m')  # ,resolution='10m'
            else:
                print('impossible to have GRD product')

        #####
        maphandler = figure(x_range=(-19000000, 8000000), y_range=(-1000000, 7000000),
                            x_axis_type="mercator", y_axis_type="mercator", plot_height=800,
                            plot_width=950, tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover,tap")
        hover = maphandler.select(dict(type=HoverTool))
        # hover.tooltips = tooltips
        maphandler.xgrid.grid_line_color = None
        maphandler.ygrid.grid_line_color = None
        # tile_provider = get_provider(CARTODBPOSITRON)
        # maphandler.add_tile(tile_provider)
        if burst_type == 'intra':
            ds = self.ds_intra
            cds = self.cds_intra
        else:
            ds = self.ds_inter
            cds = self.cds_inter
        maphandler = self.display_intra_inter_burst_grids()

        if self.xsarobjgrd:
            maphandler = self.display_roughness_grd() * maphandler  # cannot click on the point of the xspec grid
            # maphandler = maphandler*self.display_roughness_grd()
            pass
        posxy = hv.streams.Tap(source=maphandler, x=0, y=0)
        # interactive selection of a point on the map
        # Declare Tap stream with heatmap as source and initial values

        lons = cds['longitude'].values
        lats = cds['latitude'].values

        def tap_update_xspec_figures(x, y):
            if self.burst_type == 'intra':
                ds = self.ds_intra
                cds = self.cds_intra
            else:
                ds = self.ds_inter
                cds = self.cds_inter
            selected_pt = np.argmin((x - lons) ** 2 + (y - lats) ** 2)
            self.previous_xspec_selected = copy.copy(self.latest_click)
            self.latest_click = selected_pt
            print('selected_pt', selected_pt)
            burst = cds['burst'].iloc[selected_pt]
            tile_line = cds['Tline'].iloc[selected_pt]
            tile_sample = cds['Tsample'].iloc[selected_pt]
            xsrehandler1 = display_xspec_cart_holo(ds, bu=burst, li=tile_line, sam=tile_sample, typee='Re')
            xsimhandler1 = display_xspec_cart_holo(ds, bu=burst, li=tile_line, sam=tile_sample, typee='Im')
            # rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
            #                         tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
            rough_handler1 = self.display_roughness_slc(L1B_file, subswath_id, burst=burst, tile_sample_id=tile_sample,
                                                        tile_line_id=tile_line,
                                                        dsl1b=ds)

            burst_prev = cds['burst'].iloc[self.previous_xspec_selected]
            tile_line_prev = cds['Tline'].iloc[self.previous_xspec_selected]
            tile_sample_prev = cds['Tsample'].iloc[self.previous_xspec_selected]
            xsrehandler2 = display_xspec_cart_holo(ds, bu=burst_prev, li=tile_line_prev, sam=tile_sample_prev,
                                                   typee='Re')
            xsimhandler2 = display_xspec_cart_holo(ds, bu=burst_prev, li=tile_line_prev, sam=tile_sample_prev,
                                                   typee='Im')
            # rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
            #                         tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
            rough_handler2 = self.display_roughness_slc(L1B_file, subswath_id, burst=burst_prev,
                                                        tile_sample_id=tile_sample_prev,
                                                        tile_line_id=tile_line_prev,
                                                        dsl1b=ds)
            res = pn.Column(
                pn.Row(xsrehandler1, xsimhandler1, rough_handler1),
                # layout_1,
                pn.Row(xsrehandler2, xsimhandler2, rough_handler2),
            )
            return res

        # Connect the Tap stream to the tap_histogram callback
        # tap_dmap = hv.DynamicMap(tap_update_xspec_figures, streams=[posxy, checkbox])
        layout_figures = pn.Row(pn.bind(tap_update_xspec_figures, x=posxy.param.x,
                                        y=posxy.param.y))
        bokekjap = pn.Row(
            pn.Column(pn.Column(checkbox_files, pn.Row(checkbox_burst, checkbox_subswath), maphandler, posxy)),
            layout_figures,
            # pn.Column(
            # pn.Row(xsrehandler1,xsimhandler1,rough_handler1),

            # layout_1,
            # pn.Row(xsrehandler2, xsimhandler2, rough_handler2),
            # )
        )
        return bokekjap


############################################################################
#####################NO MAIN BUT IT LOOKS LIKE#############################
small_plot_height = 300
small_plot_with = 400

xsrehandler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsrehandler1 = display_xspec_cart_holo(ds_intra,bu = 3,li = 0,sam = 4,typee='Re')
xsimhandler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsimhandler1 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 4,typee='Im')
rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                        tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")

xsrehandler2 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsrehandler2 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 5,typee='Re')
xsimhandler2 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsimhandler2 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 5,typee='Im')
rough_handler2 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
                        tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")

# button intra / inter
checkbox_burst = pn.widgets.Select(options=['intra', 'inter'], name='burst Type')
checkbox_subswath = pn.widgets.Select(options=['iw1_vv', 'iw2_vv', 'iw3_vv', 'iw1_vh'], name='subswath #')
streams_select = dict(burst_type=checkbox_burst.param.value)
# all_avail_l1B = glob.glob('/home1/scratch/agrouaze/l1b/S1*/s1*L1B_xspec_IFR*0.6*.nc')
all_avail_l1B = glob.glob(
    '/home/datawork-cersat-public/project/sarwave/data/products/tests/v7/S1*/s1*L1B_xspec_IFR*0.6*.nc')
print('all available L1B', len(all_avail_l1B))
checkbox_files = pn.widgets.Select(options=all_avail_l1B, name='file')
# files = pn.widgets.FileSelector('/home1/scratch/agrouaze/l1b/',only_files=True,file_pattern='S1*L1B_xspec_IFR*.nc') # ,value=[L1B_file_default]
# logging.info('defaut component defined')
# widget_dmap = hv.DynamicMap(update_app_burst, streams=streams_select)
# widget_dmap = pn.bind(update_app_burst,burst_type=checkbox.param.value)
instclass = monAppIW_SLC()
layout = pn.Row(pn.bind(instclass.update_app_burst,
                        burst_type=checkbox_burst.param.value,
                        subswath_id=checkbox_subswath.param.value,
                        L1B_file=checkbox_files.param.value
                        ))
# pn.Row(widget_dmap).servable()
# layout = pn.Row(xsrehandler1)
layout.servable()
# pn.panel("# Test").servable()
