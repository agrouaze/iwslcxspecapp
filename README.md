# iwslcxspecapp
![image](https://user-images.githubusercontent.com/14925067/228569320-2013fb23-9ca9-40ce-8a1d-912347ceeea9.png)

you need to display the content of L1B IFREMER products? 

# install
this web app relies on some libraries:
 - holoviews
 - datashader
 - hvplot
 - pandas
 - xarray
 - xsar
 - cartopy

installation is only presents to help solving the dependencies:


```bash
pip install .
```

# start web app 

```bash
bash ./scripts/start_app.bash
```

checkout the URL displayed in stdout (e.g. https://localhost:5007 )

# config [optional]
If you want to change the location of L1B and L1 SLC and/or GRD products

edit file `iwslcxspecapp/config.yml`
