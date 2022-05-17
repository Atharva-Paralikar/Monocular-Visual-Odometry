## License
```
MIT License

Copyright (c) 2022 Atharva Paralikar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## Author
Atharva Paralikar - M.Engg Robotics Student at University of Maryland, College Park

Hemanth Joseph Raj - M.Engg Robotics Student at University of Maryland, College Park
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
