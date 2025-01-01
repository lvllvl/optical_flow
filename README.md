# Optical Flow

Optical flow from scratch

## Introduction

To use this install requirements: 
`pip install -r requirements.txt`


### Running the script

You can just run the script with all the defaults

`pthon main.py`

Or you can customize using the following flags, here's an example:

`pytnon main.py --input ./data/another_video.mpy --num-levels 5 --window-size 9 --output-dir ./my_outputs`

- `--input`
    - type=`str`
    - default = `./data/test_countryroad.mp4` 
- `--num-levels`
    - type=int
    - default = 3
- `--output-dir`
    - type = str
    - default = "./outputs/"
