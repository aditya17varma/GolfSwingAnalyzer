# GolfSwingAnalyzer
Compare your golf swing with PGA pros!

## To Run
You need a video file of your golf swing. Either from the front or side persepctives. For best perfomance with GolfDB event detection, the video should start a couple of seconds before the backswing starts.

Place the video file in the input folder.

The main function is in PoseDetection/src/analyzer.py.

To run the main, provide the path to the input video and the perspective.

For example:
```
python3 -m analyzer -i ../input/inputVideo -p Side
```

or 

```
python3 -m analyzer -i ../input/inputVideo -p Front
```

If there is an issue with the GolfDB submodule not being found, you may need to create a soft link:
```
ln -s ../../submodule/GolfDB GolfDB
```

