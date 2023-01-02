#!/bin/bash
# creation: January 2023
# gogogo
hosti=`hostname`
echo 'start panel python app to visualize IW SLC L1B x-spectrum'
panel serve ../iwslcxspecapp/iw_slc_xspec_visu_panel_app.py --allow-websocket-origin=$hosti:5007 --port=5007 --autoreload