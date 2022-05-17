## Initial Steps:

1. Copy the repository
```
git clone --recursive https://github.com/Atharva-Paralikar/Monocular-Visual-Odometry
```
2. Source the repository 
```
cd ~/Monocular-Visual-Odometry
```

A) If the dataset is a video

1. Place the video in the root folder.

2. Run the command

```
python3 createframes.py
```
B) If the dataset is set of distroted frames

1. Place the frames inside the folder "./dataset/Oxford_dataset/stereo/centre"
and place the lookup table information in the "mono" folder

2. Run the command

```
python3 processdata.py
```

C) If the dataset is set of undistorted frames

1. Place the frames inside the folder "./dataset/processedframes"


## Running the code

1. Source the repository 
```
cd ~/Monocular-Visual-Odometry
```
2. Run the command

```
python3 vo_sift.py
```


Output link
https://drive.google.com/drive/folders/18yW_rZ-sqJoPXj6TFAKzS_p4mec82GMU?usp=sharing