import glob
import os
import panel as pn
import logging
from bokeh.plotting import figure
from iwslcxspecapp.iw_slc_xspec_visu_panel_app import monAppIW_SLC,small_plot_width,\
    small_plot_height,checkbox_burst,checkbox_subswath
import yaml
# Read YAML file
src_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)),'iwslcxspecapp'))
if os.path.exists(os.path.join(src_dir,"localconfig.yml")):
    config_path = os.path.join(src_dir,"localconfig.yml")
else:
    config_path = os.path.join(src_dir,"config.yml")

print('config_path',config_path)
with open(config_path, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
# L1B_file_default = "/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20180104T061957_20180104T062025_020001_02211F_77E6.SAFE_L1B_xspec_IFR_VV_0.1.nc"
# L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20210413T043041_20210413T043109_037427_04695F_FE5C.SAFE_L1B_xspec_IFR_0.3.nc'
# L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20220921T063225_20220921T063252_045099_05639C_D284.SAFE/s1a-iw2-slc-vv-20220921t063225-20220921t063251-045099-05639c-005_L1B_xspec_IFR_0.6.nc'
# L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20220921T063225_20220921T063252_045099_05639C_D284.SAFE/s1a-iw2-slc-vv-20220921t063225-20220921t063251-045099-05639c-005_L1B_xspec_IFR_0.6.nc'
# L1B_file_default = '/home1/scratch/agrouaze/l1b/S1A_IW_SLC__1SDV_20170907T103019_20170907T103047_018268_01EB76_5F55.SAFE/s1a-iw2-slc-vv-20170907t103020-20170907t103045-018268-01eb76-005_L1B_xspec_IFR_0.6.nc'
# subswath = 'subswath_1'
#subswath = 'iw1_vv'
L1B_file_default = data_loaded['datadir']
files_dir = data_loaded['files_dir']
if files_dir == 'None':
    files_dir = os.path.abspath(os.path.join(src_dir,'..','assets'))
print('L1B_file_default',L1B_file_default)



############################################################################
#####################NO MAIN BUT IT LOOKS LIKE#############################
logging.basicConfig(level=logging.INFO,force=True)

xsrehandler1 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsrehandler1 = display_xspec_cart_holo(ds_intra,bu = 3,li = 0,sam = 4,typee='Re')
xsimhandler1 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsimhandler1 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 4,typee='Im')
rough_handler1 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                        tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")

xsrehandler2 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsrehandler2 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 5,typee='Re')
xsimhandler2 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                      tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")
# xsimhandler2 = display_xspec_cart_holo(ds,bu = 3,li = 0,sam = 5,typee='Im')
rough_handler2 = figure(plot_height=small_plot_height, plot_width=small_plot_width,
                        tools="pan, wheel_zoom, box_zoom, reset,lasso_select,hover")

# button intra / inter

streams_select = dict(burst_type=checkbox_burst.param.value)

#files_dir = os.path.abspath(os.path.join(src_dir,'..','assets','S1*','s1*L1B_xspec_IFR*.nc'))
pattern_list_files_dir = os.path.abspath(os.path.join(files_dir,'S1*','s1*L1B_xspec_IFR*.nc'))
print('files_dir',files_dir)
print('pattern_list_files_dir',pattern_list_files_dir)
all_avail_l1B = sorted(glob.glob(pattern_list_files_dir))
print('all available L1B', len(all_avail_l1B))

# files = pn.widgets.FileSelector('/home1/scratch/agrouaze/l1b/',only_files=True,file_pattern='S1*L1B_xspec_IFR*.nc') # ,value=[L1B_file_default]
# logging.info('defaut component defined')
# widget_dmap = hv.DynamicMap(update_app_burst, streams=streams_select)
# widget_dmap = pn.bind(update_app_burst,burst_type=checkbox.param.value)
instclass = monAppIW_SLC()

layout = pn.Row(pn.bind(instclass.update_app_burst,
                        burst_type=checkbox_burst.param.value,
                        subswath_id=checkbox_subswath.param.value,
                        L1B_file=instclass.get_checkboxes(all_avail_l1B).param.value,
                        all_avail_l1B = all_avail_l1B,
                        ))
# pn.Row(widget_dmap).servable()
# layout = pn.Row(xsrehandler1)
print('go servable')
layout.servable()
# pn.panel("# Test").servable()
