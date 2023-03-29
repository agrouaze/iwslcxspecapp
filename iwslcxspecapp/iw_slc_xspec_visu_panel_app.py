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
import geopandas as gpd
import copy
import hvplot.pandas
import hvplot.xarray  # noqa
import cartopy.crs as ccrs
import geoviews as gv
import xsar
from shapely.ops import transform
from shapely.geometry import Point
import pyproj
import sys
# import iwslcxspecapp.get_path_from_base_SAFE
# import iwslcxspecapp.match_GRD_SLC
from iwslcxspecapp.symmetrize_xspec import symmetrize_xspectrum
import os
import glob
from cartopy import crs
from holoviews.operation.datashader import datashade, rasterize
from matplotlib import colors as mcolors
import pandas as pd
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter,
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
from bokeh.plotting import figure
webMercator = pyproj.CRS('EPSG:3857')
main_map_width = 600
main_map_height = 500
small_plot_height = 300
small_plot_width = 400
checkbox_burst = pn.widgets.Select(options=['intra', 'inter'], name='burst Type')
checkbox_subswath = pn.widgets.Select(options=['iw1_vv', 'iw2_vv', 'iw3_vv', 'iw1_vh'], name='subswath #')


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
    logging.debug('ds[sample] %s', ds['sample'])
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
    #dt = datatree.open_datatree(L1B_file)
    ds = xr.open_dataset(L1B_file, group=typee+'burst')
    # if typee == 'intra':
    #     # ds = dt[subswath]['intraburst_xspectra'].to_dataset()
    #     #ds = dt['intraburst'].to_dataset()
    #
    # else:
    #     # ds = dt[subswath]['interburst_xspectra'].to_dataset()
    #     #ds = dt['interburst'].to_dataset()
    return ds


#
def get_tabulated_burst_data(ds):
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
    logging.debug('start tabulation L1B')
    #print('ds["corner_longitude"]', ds['corner_longitude'])
    for iburst in range(ds.burst.size):
        for itilesample in range(ds['corner_longitude'].tile_sample.size):
            for itileline in range(ds['corner_longitude'].tile_line.size):
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
    logging.debug('start display_xspec_cart_holo')
    if typee == 'Re':
        cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "violet", "mediumpurple", "cyan", "springgreen",
                                                              "yellow", "red"])
    else:
        cmap = 'PuOr'

    # if False:
    #     set_xspec = set_xspec.assign_coords({'ky': set_xspec['k_az'], 'kx': set_xspec['k_rg']})
    #     set_xspec = set_xspec.swap_dims({'freq_line': 'ky', 'freq_sample': 'kx'})
    #     kx_varname = 'kx'
    #     ky_varname = 'ky'
    # else:
    # gather Re and Imaginary part
    if 'xspectra_0tau_Re' in ds: #intra burst case
        for tautau in range(3):
            ds['xspectra_%stau' % tautau] = ds['xspectra_%stau_Re' % tautau] + 1j * ds['xspectra_%stau_Im' % tautau]
            ds = ds.drop(['xspectra_%stau_Re' % tautau, 'xspectra_%stau_Im' % tautau])
    else: # inter burst case
        ds['xspectra'] = ds['xspectra_Re'] + 1j * ds['xspectra_Im']
        ds = ds.drop(['xspectra_Re', 'xspectra_Im'])
    set_xspec = ds[{'burst': bu, 'tile_line': li, 'tile_sample': sam}]
    set_xspec = set_xspec.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
    logging.debug('symmetrize_xspectrum')
    set_xspec = symmetrize_xspectrum(set_xspec, dim_range='k_rg', dim_azimuth='k_az')
    logging.debug('symmetrzied')
    kx_varname = 'k_rg'
    ky_varname = 'k_az'
    # sp = set_xspec['xspectra_2tau_%s' % typee].mean(dim='2tau')
    if 'xspectra_2tau' in set_xspec:
        sp = set_xspec['xspectra_2tau'].mean(dim='2tau')
    else:
        sp = set_xspec['xspectra'] # inter burst case
    sp2 = sp.where(np.logical_and(np.abs(sp[kx_varname]) <= 0.14, np.abs(sp[ky_varname]) <= 0.14), drop=True)
    logging.debug('spectra is ready')
    hv.extension('bokeh')
    if typee == 'Re':
        im = hv.Image(abs(sp2.real), kdims=[kx_varname, ky_varname]).opts(width=small_plot_width,
                                                                          height=small_plot_height,
                                                                          cmap=cmap, colorbar=True,
                                                                          xlim=(    -0.07, 0.07),
                                                                          ylim=(-0.07, 0.07),
                                                                          )  # gist_ncar_r , 'cet_linear_wyor_100_45_c55'
    else:
        if np.all(np.isnan(sp2.imag)):
            extrema = 1
        else:
            extrema = float(abs(sp2.imag).max().values)

        im = hv.Image(sp2.imag, kdims=[kx_varname, ky_varname]).opts(width=small_plot_width,
                                                                     height=small_plot_height,
                                                                     cmap=cmap, colorbar=True,
                                                                     xlim=(-0.07, 0.07),
                                                                     ylim=(-0.07, 0.07),
                                                                     show_grid=True, clim=(-extrema, extrema))
    tt = hv.Text(-0.06, 0.05, 'burst %s\ntile range: %s\ntile azimuth: %s' % (bu, sam, li), halign='left', fontsize=8)
    logging.debug('add circles on plot')
    cc = add_cartesian_wavelength_circles(default_values=[100, 300, 600])
    return im * tt * cc
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
        self.lons =  None
        self.lats = None



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
        logging.debug('start roughness display')
        lines, samples = get_tile_corner_image_idx(dsl1b, bursti=burst,
                                                   tile_line_i=tile_line_id,
                                                   tile_sample_i=tile_sample_id)
        logging.debug('lines %s', lines)
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
        rough = self.xsarobjgrd.dataset['sigma0'].rio.reproject('epsg:4326',
                                                                shape=(1000, 1000), nodata=np.nan).isel({'pol': 0})
        logging.debug(rough)
        # rough = abs(self.xsarobjgrd.dataset['sigma0'].isel({ 'pol': 0})).values.ravel()

        p95percentile = np.nanpercentile(rough.squeeze().values.ravel(), 95)
        logging.debug('p95percentile GRD %s', p95percentile)
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

    def set_input_l1b_data(self,L1B_file,burst_type):
        self.l1bpath = L1B_file
        subswath_id = os.path.basename(L1B_file).split('-')[1] + '_' + os.path.basename(L1B_file).split('-')[3]
        self.subswath = subswath_id
        self.burst_type = burst_type
        self.ds_intra = read_data_L1B(L1B_file, typee='intra')
        self.cds_intra = get_tabulated_burst_data(self.ds_intra)
        self.ds_inter = read_data_L1B(L1B_file, typee='inter')
        self.cds_inter = get_tabulated_burst_data(self.ds_inter)

    @pn.depends(checkbox_burst.param.value)
    def reset_index_xpsec_selected(self):
        """
        index of points selected are integer in the order of vectorized tabulated data
        :return:
        """
        self.previous_xspec_selected = 0
        self.latest_click = 0


    def update_app_burst(self,burst_type=None, subswath_id=None, L1B_file=None, all_avail_l1B=None):
        """

        Parameters
        ----------
        burst_type:  str intra or inter
        subswath_id: str iw1
        L1B_file: str one subswath .nc file (previous version it was one nc file for a SAFE)
        all_avail_l1B : list

        Returns
        -------

        """
        logging.info('update_app_burst')
        self.display_rough = False  # tmp swith off for local test, march 2023
        # if L1B_file==[]:
        #     L1B_file = L1B_file_default #security util???
        if isinstance(L1B_file, list):
            if len(L1B_file) != 0:
                L1B_file = L1B_file[0]
                logging.debug('new L1B %s ', L1B_file)
            else:
                pass
                # L1B_file = L1B_file_default  # security util???
        # prepare data
        logging.info('oui')

        logging.debug('L1B_file %s', L1B_file)
        if L1B_file != self.l1bpath or subswath_id != self.subswath or burst_type != self.burst_type:
            logging.debug('go for reading L1B')
            self.set_input_l1b_data(L1B_file,burst_type)

            logging.debug('ok data is loaded')
            if self.display_rough:
                # base = os.path.basename(self.l1bpath).split('_L1B')[0]
                base = os.path.basename(os.path.dirname(self.l1bpath))
                logging.debug('base %s', base)
                # fullpath_safeL1SLC = get_path_from_base_SAFE.get_path_from_base_SAFE(base)
                fullpath_safeL1SLC = os.path.join(os.path.dirname(os.path.dirname(L1B_file)), 'raw_data', base)
                logging.debug('fullpath_safeL1SLC %s', fullpath_safeL1SLC)
                if subswath_id is not None:
                    subswath_nb = subswath_id.split('_')[0][-1]
                str_gdal = 'SENTINEL1_DS:%s:IW%s' % (fullpath_safeL1SLC, subswath_nb)
                logging.debug('str_gdal %s', str_gdal)
                self.xsarobjslc = xsar.Sentinel1Dataset(str_gdal)  # ,resolution='10m'
                grdh_path = match_GRD_SLC.match_SLC_GRD(os.path.basename(fullpath_safeL1SLC), type_seek='GRDH')
                if grdh_path:
                    self.xsarobjgrd = xsar.Sentinel1Dataset(grdh_path, resolution='200m')  # ,resolution='10m'
                else:
                    logging.debug('impossible to have GRD product')
        else:
            logging.debug('nothing to do')

        #####
        self.maphandler = figure(x_range=(-19000000, 8000000), y_range=(-1000000, 7000000),
                         x_axis_type="mercator", y_axis_type="mercator", plot_height=800,
                         plot_width=950, tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover,tap")
        hover = self.maphandler.select(dict(type=HoverTool))
        # hover.tooltips = tooltips
        self.maphandler.xgrid.grid_line_color = None
        self.maphandler.ygrid.grid_line_color = None
        # tile_provider = get_provider(CARTODBPOSITRON)
        # maphandler.add_tile(tile_provider)
        if burst_type == 'intra':
            ds = self.ds_intra
            cds = self.cds_intra
        else:
            ds = self.ds_inter
            cds = self.cds_inter
        logging.debug('start grid display')
        self.maphandler = self.display_intra_inter_burst_grids()

        if self.xsarobjgrd:
            logging.debug('display rougness grid')
            self.maphandler = self.display_roughness_grd() * self.maphandler  # cannot click on the point of the xspec grid
            # maphandler = maphandler*self.display_roughness_grd()
            pass
        posxy = hv.streams.Tap(source=self.maphandler, x=0, y=0)
        logging.debug('posxy %s %s %s',posxy,type(posxy),dir(posxy))
        # interactive selection of a point on the map
        # Declare Tap stream with heatmap as source and initial values





        # Connect the Tap stream to the tap_histogram callback
        # tap_dmap = hv.DynamicMap(tap_update_xspec_figures, streams=[posxy, checkbox])
        logging.debug('creating rows and columns for layout panel/bokeh')
        logging.debug('posxy.param.x %s %s %s', posxy.param.x, type(posxy.param.x), dir(posxy.param.x))
        layout_figures = pn.Row(pn.bind(self.tap_update_xspec_figures, x=posxy.param.x,
                                        y=posxy.param.y))


        checkbox_files = self.get_checkboxes(all_avail_l1B=all_avail_l1B)
        bokekjap = pn.Row(
            pn.Column(pn.Column(checkbox_files, pn.Row(checkbox_burst, checkbox_subswath),
                                self.maphandler)),
            layout_figures,
        )
        # bokekjap = pn.Row(
        #     pn.Column(pn.Column(checkbox_files, bindo_burst,pn.Row(checkbox_subswath),
        #                         self.maphandler, posxy)),
        #     layout_figures,
        # )
        logging.debug('return bokeh app layout')
        return bokekjap

    def tap_update_xspec_figures(self,x, y):
        """
        :param y:
        :return:
        """
        if self.burst_type == 'intra':
            ds = self.ds_intra
            cds = self.cds_intra
        else:
            ds = self.ds_inter
            cds = self.cds_inter

        project = pyproj.Transformer.from_crs(webMercator, self.projection, always_xy=True).transform

        xygeo = transform(project, Point(x,y))
        logging.debug('xny %s %s',x,y)
        logging.debug('xygeo, %s %s %s',xygeo, xygeo.x, xygeo.y)

        selected_pt = np.argmin((xygeo.x - self.lons) ** 2 + (xygeo.y - self.lats) ** 2)
        self.previous_xspec_selected = copy.copy(self.latest_click)
        self.latest_click = selected_pt
        logging.debug('selected_pt %s', selected_pt)
        burst = cds['burst'].iloc[selected_pt]
        tile_line = cds['Tline'].iloc[selected_pt]
        tile_sample = cds['Tsample'].iloc[selected_pt]
        xsrehandler1 = display_xspec_cart_holo(ds, bu=burst, li=tile_line, sam=tile_sample, typee='Re')
        xsimhandler1 = display_xspec_cart_holo(ds, bu=burst, li=tile_line, sam=tile_sample, typee='Im')
        # rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
        #                         tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
        logging.debug('xspec figures are OK')
        if self.display_rough:
            rough_handler1 = self.display_roughness_slc(self.l1bpath, self.subswath, burst=burst,
                                                        tile_sample_id=tile_sample,
                                                        tile_line_id=tile_line,
                                                        dsl1b=ds)
        else:
            logging.debug('empty roughness figure')
            rough_handler1 = hv.Image((np.random.rand(100, 100)))
            logging.debug('done')

        logging.debug('start to create 2nd set of xspec figures')
        burst_prev = cds['burst'].iloc[self.previous_xspec_selected]
        tile_line_prev = cds['Tline'].iloc[self.previous_xspec_selected]
        tile_sample_prev = cds['Tsample'].iloc[self.previous_xspec_selected]

        xsrehandler2 = display_xspec_cart_holo(ds, bu=burst_prev, li=tile_line_prev, sam=tile_sample_prev,
                                               typee='Re')
        xsimhandler2 = display_xspec_cart_holo(ds, bu=burst_prev, li=tile_line_prev, sam=tile_sample_prev,
                                               typee='Im')
        # rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_with,
        #                         tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
        if self.display_rough:
            rough_handler2 = self.display_roughness_slc(self.l1bpath, self.subswath, burst=burst_prev,
                                                        tile_sample_id=tile_sample_prev,
                                                        tile_line_id=tile_line_prev,
                                                        dsl1b=ds)
        else:
            rough_handler2 = hv.Image(np.random.rand(100, 100))
        res = pn.Column(
            pn.Row(xsrehandler1, xsimhandler1, rough_handler1),
            # layout_1,
            pn.Row(xsrehandler2, xsimhandler2, rough_handler2),
        )
        #another call to re draw the map and change the red circle position
        #self.maphandler = self.display_intra_inter_burst_grids() #-> leads to not clickable map... to debug
        return res

    def get_checkboxes(self, all_avail_l1B):
        ##############################
        # add the checboxes

        checkbox_files = pn.widgets.Select(options=all_avail_l1B, name='file')
        ##############################
        return checkbox_files


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
            ds = self.ds_intra.load()
            cds = self.cds_intra
        else:
            ds = self.ds_inter.load()
            cds = self.cds_inter
        self.lons = cds['longitude'].values
        self.lats = cds['latitude'].values
        ds['corner_longitude'] = ds['corner_longitude']
        ds['corner_latitude'] = ds['corner_latitude']
        for iburst in range(ds.burst.size):
            for itilesample in range(ds['corner_longitude'].tile_sample.size):
                for itileline in range(ds['corner_longitude'].tile_line.size):
                    clon = ds['corner_longitude'].isel(
                        {'burst': iburst, 'tile_line': itileline,
                         'tile_sample': itilesample}).values.ravel(order='A')
                    clat = ds['corner_latitude'].isel(
                        {'burst': iburst, 'tile_line': itileline,
                         'tile_sample': itilesample}).values.ravel(order='A')
                    clon2 = copy.copy(clon)
                    clat2 = copy.copy(clat)
                    clat2[2] = clat[3]
                    clat2[3] = clat[2]
                    clon2[2] = clon[3]
                    clon2[3] = clon[2]
                    tmpo = np.stack([clon2, clat2]).T
                    tmpo = np.vstack([tmpo, tmpo[0, :]])
                    tmppoly = gv.Path((tmpo[:, 0], tmpo[:, 1]), kdims=['Longitude', 'Latitude'], ).opts(color='grey')
                    all_poly.append(tmppoly)
        logging.debug('polygons are constructed')
        self.projection = ccrs.PlateCarree()
        if self.burst_type == 'intra':
            coco = 'blue'
        else:
            coco = 'red'
        points = cds.hvplot.points(x='longitude', y='latitude', hover_cols='all', use_index=False,
                                   label=self.burst_type, color=coco, geo=True,
                                   crs=self.projection).opts(tools=['hover','tap'],
                                                             size=10,nonselection_alpha=0.1)
        logging.debug('crs %s', points.crs)
        # points = gv.Points(('longitude','latitude'),source=cds).opts(tools=['hover'])
        # gv.tile_sources.Wikipedia  *


        res = (gv.tile_sources.EsriImagery * gv.Overlay(all_poly) * points).opts(width=main_map_width,
                                                                                         height=main_map_height,
                                                                                         show_legend=True,
                                                                                         title=os.path.basename(
                                                                                             self.l1bpath) + '\n' + self.subswath,
                                                                                         fontsize={'title': 8})
        return res

############################################################################
#####################NO MAIN BUT IT LOOKS LIKE#############################
