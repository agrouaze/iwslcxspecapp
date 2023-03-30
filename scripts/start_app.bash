#!/bin/bash
# creation: January 2023
# gogogo
hosti=`hostname`
echo 'start panel python app to visualize IW SLC L1B x-spectrum'
panel serve ./start_app.py --allow-websocket-origin=`hostname`:5007 --allow-websocket-origin=localhost:5007 --port=5007 --autoreload
