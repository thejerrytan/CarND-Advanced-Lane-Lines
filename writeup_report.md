**Advanced Lane Finding Project**

By: Jerry Tan Si Kai

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[binary_thresholding_uwarped_with_polygon_straight_lines1]: ./output_images/binary_thresholding_unwarped_with_polygon_straight_lines1.jpg "binary_thresholding_uwarped_with_polygon_straight_lines1"
[binary_thresholding_unwarped_with_polygon_straight_lines2]: ./output_images/binary_thresholding_unwarped_with_polygon_straight_lines2.jpg "binary_thresholding_unwarped_with_polygon_straight_lines2"
[binary_thresholding_unwarped_with_polygon_test1]: ./output_images/binary_thresholding_unwarped_with_polygon_test1.jpg "binary_thresholding_unwarped_with_polygon_test1"
[binary_thresholding_unwarped_with_polygon_test2]: ./output_images/binary_thresholding_unwarped_with_polygon_test2.jpg "binary_thresholding_unwarped_with_polygon_test2"
[binary_thresholding_unwarped_with_polygon_test3]: ./output_images/binary_thresholding_unwarped_with_polygon_test3.jpg "binary_thresholding_unwarped_with_polygon_test3"
[binary_thresholding_unwarped_with_polygon_test4]: ./output_images/binary_thresholding_unwarped_with_polygon_test4.jpg "binary_thresholding_unwarped_with_polygon_test4"
[binary_thresholding_unwarped_with_polygon_test5]: ./output_images/binary_thresholding_unwarped_with_polygon_test5.jpg "binary_thresholding_unwarped_with_polygon_test5"
[binary_thresholding_unwarped_with_polygon_test6]: ./output_images/binary_thresholding_unwarped_with_polygon_test6.jpg "binary_thresholding_unwarped_with_polygon_test6"
[binary_thresholding_warped_straight_lines1]: ./output_images/binary_thresholding_warped_straight_lines1.jpg "binary_thresholding_warped_straight_lines1"
[binary_thresholding_warped_straight_lines2]: ./output_images/binary_thresholding_warped_straight_lines2.jpg "binary_thresholding_warped_straight_lines2"
[binary_thresholding_warped_test1]: ./output_images/binary_thresholding_warped_test1.jpg "binary_thresholding_warped_straight_lines1"
[binary_thresholding_warped_test2]: ./ouptut_images/binary_thresholding_warped_test2.jpg "binary_thresholding_warped_test3"
[binary_thresholding_warped_test3]: ./output_images/binary_thresholding_warped_test3.jpg "binary_thresholding_warped_test3"
[binary_thresholding_warped_test4]: ./output_images/binary_thresholding_warped_test4.jpg "binary_thresholding_warped_test4"
[binary_thresholding_warped_test5]: ./output_images/binary_thresholding_warped_test5.jpg "binary_thresholding_warped_test5"
[binary_thresholding_warped_test6]: ./output_images/binary_thresholding_warped_test6.jpg "binary_thresholding_warped_test6"
[sliding_window_test1]: ./output_images/sliding_window_test1.jpg "sliding_window_test1"
[sliding_window_test2]: ./output_images/sliding_window_test2.jpg "sliding_window_test2"
[sliding_window_test3]: ./output_images/sliding_window_test3.jpg "sliding_window_test3"
[sliding_window_test4]: ./output_images/sliding_window_test4.jpg "sliding_window_test4"
[sliding_window_test5]: ./output_images/sliding_window_test5.jpg "sliding_window_test5"
[sliding_window_test6]: ./output_images/sliding_window_test6.jpg "sliding_window_test6"
[sliding_window_straight_lines1]: ./output_images/sliding_window_straight_lines1.jpg "sliding_window_straight_lines1"
[sliding_window_straight_lines2]: ./output_images/sliding_window_straight_lines2.jpg "sliding_window_straight_lines2"
[undistorted1]: ./output_images/undistorted1.jpg "undistorted1"
[undistorted2]: ./output_images/undistorted2.jpg "undistorted2"
[undistorted3]: ./output_images/undistorted3.jpg "undistorted3"
[undistorted4]: ./output_images/undistorted4.jpg "undistorted4"
[undistorted5]: ./output_images/undistorted4.jpg "undistorted5"
[undistorted6]: ./output_images/undistorted6.jpg "undistorted6"
[undistorted7]: ./output_images/undistorted7.jpg "undistorted7"
[undistorted8]: ./output_images/undistorted8.jpg "undistorted8"
[undistorted9]: ./output_images/undistorted9.jpg "undistorted9"
[undistorted10]: ./output_images/undistorted10.jpg "undistorted10"
[undistorted11]: ./output_images/undistorted11.jpg "undistorted11"
[undistorted12]: ./output_images/undistorted12.jpg "undistorted12"
[undistorted13]: ./output_images/undistorted13.jpg "undistorted13"
[undistorted14]: ./output_images/undistorted14.jpg "undistorted14"
[undistorted15]: ./output_images/undistorted15.jpg "undistorted15"
[undistorted16]: ./output_images/undistorted16.jpg "undistorted16"
[undistorted17]: ./output_images/undistorted17.jpg "undistorted17"
[undistorted18]: ./output_images/undistorted18.jpg "undistorted18"
[undistorted19]: ./output_images/undistorted19.jpg "undistorted19"
[undistorted20]: ./output_images/undistorted20.jpg "undistorted20"
[undistorted_straight_lines1]: ./output_images/undistorted_straight_lines1.jpg "undistorted_straight_lines1"
[undistorted_straight_lines2]: ./output_images/undistorted_straight_lines2.jpg "undistorted_straight_lines2"
[undistorted_test1]: ./output_images/undistorted_test1.jpg "undistorted_test1"
[undistorted_test2]: ./output_images/undistorted_test2.jpg "undistorted_test2"
[undistorted_test3]: ./output_images/undistorted_test3.jpg "undistorted_test3"
[undistorted_test4]: ./output_images/undistorted_test4.jpg "undistorted_test4"
[undistorted_test5]: ./output_images/undistorted_test5.jpg "undistorted_test5"
[undistorted_test6]: ./output_images/undistorted_test6.jpg "undistorted_test6"
[unwarped_with_polygon_straight_lines1]: ./output_images/unwarped_with_polygon_straight_lines1.jpg "unwarped_with_polygon_straight_lines1"
[unwarped_with_polygon_straight_lines2]: ./output_images/unwarped_with_polygon_straight_lines2.jpg "unwarped_with_polygon_straight_lines2"
[unwarped_with_polygon_test1]: ./output_images/unwarped_with_polygon_test1.jpg "unwarped_with_polygon_test1"
[unwarped_with_polygon_test2]: ./output_images/unwarped_with_polygon_test2.jpg "unwarped_with_polygon_test2"
[unwarped_with_polygon_test3]: ./output_images/unwarped_with_polygon_test3.jpg "unwarped_with_polygon_test3"
[unwarped_with_polygon_test4]: ./output_images/unwarped_with_polygon_test4.jpg "unwarped_with_polygon_test4"
[unwarped_with_polygon_test5]: ./output_images/unwarped_with_polygon_test5.jpg "unwarped_with_polygon_test5"
[unwarped_with_polygon_test6]: ./output_images/unwarped_with_polygon_test6.jpg "unwarped_with_polygon_test6"
[warped3]: ./output_images/warped3.jpg "warped3"
[warped6]: ./output_images/warped6.jpg "warped6"
[warped7]: ./output_images/warped7.jpg "warped7"
[warped8]: ./output_images/warped8.jpg "warped8"
[warped9]: ./output_images/warped9.jpg "warped9"
[warped10]: ./output_images/warped10.jpg "warped10"
[warped11]: ./output_images/warped11.jpg "warped11"
[warped12]: ./output_images/warped12.jpg "warped12"
[warped13]: ./output_images/warped13.jpg "warped13"
[warped14]: ./output_images/warped14.jpg "warped14"
[warped15]: ./output_images/warped15.jpg "warped15"
[warped16]: ./output_images/warped16.jpg "warped16"
[warped17]: ./output_images/warped17.jpg "warped17"
[warped18]: ./output_images/warped18.jpg "warped18"
[warped19]: ./output_images/warped19.jpg "warped19"
[warped20]: ./output_images/warped20.jpg "warped20"
[warped_straight_lines1]: ./output_images/warped_straight_lines1.jpg "warped_straight_lines1"
[warped_straight_lines2]: ./output_images/warped_straight_lines2.jpg "warped_straight_lines2"
[warped_test1]: ./output_images/warped_test1.jpg "warped_test1"
[warped_test2]: ./output_images/warped_test2.jpg "warped_test2"
[warped_test3]: ./output_images/warped_test3.jpg "warped_test3"
[warped_test4]: ./output_images/warped_test4.jpg "warped_test4"
[warped_test5]: ./output_images/warped_test5.jpg "warped_test5"
[warped_test6]: ./output_images/warped_test6.jpg "warped_test6"
[original_with_lane_boundaries]: ./output_images/original_with_lane_boundaries.jpg "original_with_lane_boundaries"
[project_video_output]: ./project_video_output.mp4 "Project video output"
[project_video]: ./project_video.mp4 "Video"
[project_video_challenging]: ./challenge_video.mp4 "Challenge video"
[project_video_challenging_output]: ./challenge_video_output.mp4 "Challenge video output"
[project_video_more_challenging]: ./harder_challenge_video.mp4 "Harder challenge video"
[project_video_more_challenging_output]: ./harder_challenge_video_output.mp4 "Harder challenge video output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! All code refered to in this writeup is contained in pipeline.ipynb.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the 2nd and 3rd code cell of the IPython notebook pipeline.ipynb.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. This is done in generate_image_points(img, fname) function.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test images in camera_cal folder using the `cv2.undistort()` function and obtained this result: 

![alt text][undistorted1]
![alt text][undistorted2]
![alt text][undistorted3]
![alt text][undistorted4]
![alt text][undistorted5]
![alt text][undistorted6]
![alt text][undistorted7]
![alt text][undistorted8]
![alt text][undistorted9]
![alt text][undistorted10]
![alt text][undistorted11]
![alt text][undistorted12]
![alt text][undistorted13]
![alt text][undistorted14]
![alt text][undistorted15]
![alt text][undistorted16]
![alt text][undistorted17]
![alt text][undistorted18]
![alt text][undistorted19]
![alt text][undistorted20]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to the images in the test_images folder:
![alt text][undistorted_straight_lines1]
![alt text][undistorted_straight_lines2]
![alt text][undistorted_test1]
![alt text][undistorted_test2]
![alt text][undistorted_test3]
![alt text][undistorted_test4]
![alt text][undistorted_test5]
![alt text][undistorted_test6]

Below is the relevant code. I apply the camera matrix and distortion coefficients to cv2.undistort() function to arrive at a undistorted image.

```python
for idx, fname in enumerate(images):
    undistorted_fname = fname.split("/")[-1].replace("calibration", "undistorted")
    undist = undistort(imgs[idx], mtx, dst, undistorted_fname)
    warped_fname = undistorted_fname.replace("undistorted", "warped")
    original_with_lines, warped, M, Minv = warp(undist, warped_fname)
    if warped is not None:
        plotDiff(original_with_lines, warped, "Undistorted Image with polygon", "Warped Image")
```

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. I convert the image to HSL colorspace, then extract pixel values above 90 and below 255 in the S channel. This is very useful for picking out brightly colored lane lines. I also used a sobel edge detector on the x direction of the l-channel image and extracted pixels > 20 and < 255. I lowered my thresholds as the final lane finding algorthim was able to tolerate large amounts of noise but it definitely cannot work when there is no lane marking at all to follow. Furthermore, most of the noise would be irrelevant after perspective transform, leaving noise that is on the ground that we have to deal with. (thresholding steps at cell 5, color_and_gradient_thresholding() function).  Here's an example of my output for this step.

![alt text][binary_thresholding_unwarped_with_polygon_straight_lines1]
![alt text][binary_thresholding_unwarped_with_polygon_straight_lines2]
![alt text][binary_thresholding_unwarped_with_polygon_test1]
![alt text][binary_thresholding_unwarped_with_polygon_test2]
![alt text][binary_thresholding_unwarped_with_polygon_test3]
![alt text][binary_thresholding_unwarped_with_polygon_test4]
![alt text][binary_thresholding_unwarped_with_polygon_test5]
![alt text][binary_thresholding_unwarped_with_polygon_test6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 50 through 82 in the file `pipeline.ipynb` cell 2. It takes in the image to warp, and optionally, an array of source and destination points. If not given, it is assumed the chessboard image is used and findChessboardCorners() will be used to determine the source and destination points. For actual images, the source points are chosen to be in a rough trapezoid capturing the lane the car is on, extending towards horizon. The destination is a rectangle, which gives a bird's eye view of the lane. After trial and error, the final source and destination points are:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 235, 700      | 400, 720      | 
| 1080, 700     | 800, 720      |
| 680, 440      | 800, 0      	|
| 610, 440      | 400, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto the straight lane test images and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][unwarped_with_polygon_straight_lines1]
![alt text][warped_straight_lines1]
![alt text][unwarped_with_polygon_straight_lines2]
![alt text][warped_straight_lines2]
![alt text][unwarped_with_polygon_test1]
![alt text][warped_test1]
![alt text][unwarped_with_polygon_test2]
![alt text][warped_test2]
![alt text][unwarped_with_polygon_test3]
![alt text][warped_test3]
![alt text][unwarped_with_polygon_test4]
![alt text][warped_test4]
![alt text][unwarped_with_polygon_test5]
![alt text][warped_test5]
![alt text][unwarped_with_polygon_test6]
![alt text][warped_test6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I used most of the code from lesson (Sazan) and fit my lane lines with a 2nd order polynomial using the sliding window technique with some modifications - instead of doing the search between the left and right half of the image, i know that the lane lines are going to appear in a small neighbourhood of a few hundred pixels around 400 and 800, because that is by definition, where i have warped the left and right lane markers to be in the warped image. Any lane markings that show up beyond this window is likely to be an anomaly. This is done by the following code:

```python
# Constraint leftx and rightx base to a window between leftx_left_boundary:midpoint and midpoint:rightx_right_boundary
    leftx_left_boundary = 300
    rightx_right_boundary = 1000
    leftx_base = np.argmax(histogram[leftx_left_boundary:midpoint]) + leftx_left_boundary
    rightx_base = np.argmax(histogram[midpoint:rightx_right_boundary]) + midpoint
```

All of this is done in cell 6, function find_lane_lines() lines 3 - 118. Below are some images of the windows and lane lines drawn.

![alt text][sliding_window_straight_lines1]
![alt text][sliding_window_straight_lines2]
![alt text][sliding_window_test1]
![alt text][sliding_window_test2]
![alt text][sliding_window_test3]
![alt text][sliding_window_test4]
![alt text][sliding_window_test5]
![alt text][sliding_window_test6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #76 through #84 in my code in `pipeline.ipynb` in function find_lane_lines(). Following the tip given, I defined a conversion factor for converting between pixel coordinates and real world coordinates. Since the length of the lane covered by my warp perspective is longer (extends further towards the horizon, which contains more curvature information), i used 60m per 780 pixels conversion factor for y direction. I then find the coefficients of the best fit line by converting the data points from pixel space to real world coordinates, which allows me to calculate curvature in meters.

```python
# Define a converson in x and y for conversion from pixel space to meters
ym_per_px = 60 / 780
xm_per_px = 3.7 / 400

# Fit a second order polynomial to x,y in real world coordinates
left_fit_cr = np.polyfit(lefty*ym_per_px, leftx*xm_per_px, 2)
right_fit_cr = np.polyfit(righty*ym_per_px, rightx*xm_per_px, 2)

# Evaluate at bottom of the image
y_eval = out_img.shape[0]-1
# Calculate the new radii of curvature
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_px + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_px + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The dest (u',v') coordinates of the polynomial curve of best fit in warped pixel space is transformed back into the original src (u,v) perspective using cv2.perspectiveTransform() and the inverse matrix obtained during the warp step. I implemented this step in lines #1 through #13 in inverse_transform() function in `pipeline.ipynb`. The left lane is drawn in red, right lane in blue, the lane area in green. Here is an example of my result on a test image:

![alt text][original_with_lane_boundaries]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

My pipeline currently is not robust enough to work on the challenge videos, but here is its current performance:
Here's a [link to my challenge video result](./challenge_video_output.mp4)
Here's a [link to my harder challenge video result](./harder_challenge_video_output.mp4)
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Originally, i set the threshold on the s-channel too high, leaving me baffled as to why the lane markings would show up clearly on the s-channel image but a huge chunk of the lane would go missing after thresholding. I realized that lowering the threshold is much better than setting it too high as most of the noise is actually irrelevant after perspectiveTransform.

I was having problems with the sliding window search not being able to capture the current lane the car is driving on at the beginning and at the end of the image for images where the other lane lines, and the kerb is strongly visible as a line. It would capture parts of the other lane thinking its part of the best fit we are looking for. I had to decrease the window margin to 80. Constraining the sliding window search to search between 300:midpoint and midpoint:1000 for base leftx and rightx also helps to remove outliers, increase stability and increase the processing speed.

Currently, the pipeline does not work well for images where there are shadows on the lane, as that would lower the saturation in s-channel for the lane line. Histogram equalization as a preprocessing step might help to increase the contrast in this case. Tyre screech marks on the lanes would confuse the algorithm, resulting in false positives. Thresholding on the h and/or l channel might help to remove them. Lanes with high curvature would not be identified because our sliding window range is too small. We can try to overcome it by increasing the range, and use other methods to prevent fitting onto other lanes. Lastly, the perspective transform extends too far out into the horizon and would not be accurate for off-highway roads, especially mountainous, winding roads, or upward sloping roads like in the harder challenge video. We would need a way to determine the furthest extend of the flat plane that is the road and adjust our source and destination points dynamically.
