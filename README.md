# Bayesian tsunami forecasting

## Description of files

```terminal
.
├── README.md
├── data              :Input data
│   ├── case_ID.csv   :Scenario ID
│   ├── gauge_loc.csv :Locations of synthetic gauges
│   └── wave_seq.npy  :Data of wave sequences. Need to get from Zenodo (https://doi.org/10.5281/zenodo.12696848)
├── figures: Python scripts for generating figures
└── script            :Python scripts for a series of calculations
    ├── COND.yaml     :Calculation conditions
    ├── FORECAST.py   :Script for a tsunami prediction
    ├── MAIN.py       :Script for executing PRE.py and FORECAST.py 
    ├── PRE.py        :Script for a offline phase
    └── subfigure.py  :Subprogram
```

### Packeges
|Libraries|Versions|
|:---|:---:|
|python|3.8.10|
|numpy|1.22.1|
|simplekml|1.3.6|
|cupy|12.3.0|
|pandas|1.3.5|
|dask|2022.5.0|
|seaborn|0.11.2|
|scipy|1.8.1|
|cartopy|0.18.0|

## Instructions

### Data preparation

Obtain a file named wave_seq.npy from [Zenodo](https://doi.org/10.5281/zenodo.12696848) and place it in the data folder.
The data file stores a 3d array shaped (2342x62x2160), corresponding to the numbers of hypothetical scenarios, synthetic gauges and time steps.
The synthetic gauge configuration refer to the locations of existing observational networks; [NOWPHAS](https://www.mlit.go.jp/kowan/nowphas/ "リアルタイムナウファス（国土交通省港湾局，全国港湾海洋波浪情報網）") (red)，[DONET1](https://www.seafloor.bosai.go.jp/DONET/ "地震・津波観測監視システム：DONET") (blue), and [DONET2](https://www.seafloor.bosai.go.jp/DONET/ "地震・津波観測監視システム：DONET") (green), as shown below.

<img width="700" src="./README_images/gauges.png">

### Tsunami prediction

**1. `script/COND.yaml`: Condition settings**

- `GPU`: *bool*

  If you want to use GPU, you can set as `GPU: true`.

- `ver`: *str*

  Version of the tsunami prediction method. You can choose 'Fujita' and 'Nomura' to run a scenario superposition and the best-fit scenario detection, respectively.

  - `ROM`: *bool*

  If you do not want to conduct ROM, you can set as `ROM: false`.
  
- `nmod`: *int*

  The number of modes to realize ROM

- `ltest`: *[int] or 'all'*

  List of test scenario IDs. If you target specific test scenarios, you can replace 'all' with a list of IDs.

- `obs_window`: *[int]*

  List of time steps to output the prediction results.

**2. Run Main.py**

Run tsunami prediction by

```terminal
python3 script/MAIN.py
```

### Results

The following log flows after executing the script.

```terminal
$ python3 script/MAIN.py 
2024-02-29 20:34:47.103142
###  Preprocessing  ###
  - Initialization for preprocessing
  - Make figure for gauge arrangement (PNG)
  - Split data into training and test set
    Shape of training data (training scenarios, gauges, time steps):
      (1756, 62, 2160)
    Shape of test data (test scenarios, gauges, time steps):
      (586, 62, 2160)
  - sigular value decomposition for reduced-order modeling
    Shape of POD mode matrix (modes, modes):
      (62, 62)
    Shape of singular value vector (modes):
      (62,)
  - Make figure for contribution rates of singular values
  - Make figure for reconstruction errors vs used POD modes
2024-02-29 20:35:10.879646 

###  Forecasting  ###

2024-02-29 20:35:10.887076
Calculation starting with Fujita-version
With ROM via SVD using top 23 modes
Forecasting for scenario No.561 ("nagahama_91_00325")
  - Prediction Step: 0=0.0[min] for nagahama_91_00325
    Probable scenarios: ['nagahama_87_00328' 'nagahama_91_00026' 'nagahama_89_00156']
    weights : [0.00254603 0.00206278 0.0020399 ]
  - Prediction Step: 60=5.0[min] for nagahama_91_00325
    Probable scenarios: ['nagahama_91_00255' 'nagahama_91_00275' 'nagahama_91_00080']
    weights : [0.14705409 0.1397648  0.1067408 ]
  - Prediction Step: 120=10.0[min] for nagahama_91_00325
    Probable scenarios: ['nagahama_91_00255' 'nagahama_91_00275' 'nagahama_91_00080']
    weights : [0.21509331 0.1618069  0.15432459]
Calculation ending at 2024-02-29 20:35:16.249933
Make a waveform figure for No.561
Make a Taylor diagram for No.561
Make a figure of PDF for No.561
$ 
```

The following figures showing prediction results are save in `/result/`.

  |Taylor diagram|波形予測図（観測時間150s & 300s）|重みパラメータ分布|
  |:---:|:---:|:---:|
  |<img width="400" src="./README_images/taylor_Fujita.png">|<img width="400" src="./README_images/wave_Fujita_0120step.png">|<img width="400" src="./README_images/PDF_Fujita_0120step.png">
