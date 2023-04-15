#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt



def main():
    
	base_imgs = ["/home/ubuntu/Desktop/dicetest/truetests/4test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/5test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/6test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/12test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/13test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/16test.JPG",
	"/home/ubuntu/Desktop/dicetest/truetests/19test.JPG"
	]
	

    
	img_names2 = ["/home/ubuntu/Desktop/dicetest/faces/1.png", 
    "/home/ubuntu/Desktop/dicetest/faces/2.bmp", 
    "/home/ubuntu/Desktop/dicetest/faces/4.png",
    "/home/ubuntu/Desktop/dicetest/faces/5.png",
    "/home/ubuntu/Desktop/dicetest/faces/6.png",
    "/home/ubuntu/Desktop/dicetest/faces/7.png",
    "/home/ubuntu/Desktop/dicetest/faces/8.png",
    "/home/ubuntu/Desktop/dicetest/faces/9.png",
    "/home/ubuntu/Desktop/dicetest/faces/10.png",
    "/home/ubuntu/Desktop/dicetest/faces/11.png",
    "/home/ubuntu/Desktop/dicetest/faces/12.png",
    "/home/ubuntu/Desktop/dicetest/faces/13.png",
    "/home/ubuntu/Desktop/dicetest/faces/14.png",
    "/home/ubuntu/Desktop/dicetest/faces/15.png",
    "/home/ubuntu/Desktop/dicetest/faces/16.png",
    "/home/ubuntu/Desktop/dicetest/faces/17.png",
    "/home/ubuntu/Desktop/dicetest/faces/18.png",
    "/home/ubuntu/Desktop/dicetest/faces/19.png",
    "/home/ubuntu/Desktop/dicetest/faces/20.png"] 
    
	img_names = ["/home/ubuntu/Desktop/dicetest/faces/4.png",
    "/home/ubuntu/Desktop/dicetest/faces/5.png",
    "/home/ubuntu/Desktop/dicetest/faces/6.png",
    "/home/ubuntu/Desktop/dicetest/faces/12.png",
    "/home/ubuntu/Desktop/dicetest/faces/13.png",
    "/home/ubuntu/Desktop/dicetest/faces/16.png",
    "/home/ubuntu/Desktop/dicetest/faces/19.png",]
    
	orb = cv2.ORB_create()
	#detector = cv2.xfeatures2d.BEBLID_create(0.75)
	
	#orb.setScaleFactor(1.2)
	#orb.setNLevels(8)
	#orb.setEdgeThreshold(10)
	#orb.setPatchSize(30)
	#orb.setFastThreshold(10)
	
	for base_name in base_imgs:
		print(f"\n{base_name.split('/')[6]}-")
		base_img = cv2.imread(base_name, cv2.IMREAD_GRAYSCALE)
		baseimgkp2, baseimgdesc2 = orb.detectAndCompute(base_img, None)
		
		"""edges = cv2.Canny(base_img, 100, 200)

		# Find contours in the image
		contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Filter contours by area and shape to find the triangle
		for cnt in contours:
			
			area = cv2.contourArea(cnt)
			if area < 20 or area > 5000:
				continue
			approx = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, True), True)
			if len(approx) != 3:
				continue

			# Draw the triangle onto the output image
			base_img = cv2.drawContours(base_img, [cnt], 0, (0, 255, 0), 2)
		base_img = cv2.resize(base_img, (0, 0), fx=0.3, fy=0.3)

		# Display the output image
		cv2.imshow('Output Image', base_img)
		cv2.waitKey(0)"""
		 
		for img_name in img_names:
		
			
			faceimg = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
		
			keypoints1, descriptor1 = orb.detectAndCompute(faceimg, None)
			
			#keypoints, descriptor  = detector.compute(faceimg, None)
			
			bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
			matches2 = bf.match(descriptor1,baseimgdesc2)
			matches = sorted(matches2, key = lambda x:x.distance)	
			
			
			src_pts  = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
			dst_pts  = np.float32([baseimgkp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
			#matches = [m for i, m in enumerate(matches) if mask[i]]
			
			h,w = faceimg.shape
			pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			dst = cv2.perspectiveTransform(pts,M)
        						
			base_img2 = cv2.polylines(cv2.imread(base_name, cv2.IMREAD_GRAYSCALE), [np.int32(dst)], True, (0,0,255), 4, cv2.LINE_AA)
			
			img3 = cv2.drawMatches(
			faceimg,
			keypoints1,
			base_img2,
			baseimgkp2,
			matches,
			None,
			flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
			
			#img2 = cv2.drawKeypoints(base_img, baseimgkp, None, color=(0,255,0), flags=0)
			#result = cv2.matchTemplate(base_img, faceimg, cv2.TM_CCOEFF_NORMED)
			#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
			print(f"\n{img_name.split('/')[6]}-")
			
			#w, h = faceimg.shape[::-1]  # Get the width and height of the template
			#top_left = max_loc  # Get the location of the top-left corner of the template
			#bottom_right = (top_left[0] + w, top_left[1] + h)  # Get the location of the bottom-right corner of the template
			#cv2.rectangle(base_img, top_left, bottom_right, 255, 2)  # Draw a rectangle around the template

			base2 = cv2.resize(img3, (0, 0), fx=0.3, fy=0.3)
			plt.imshow(base2),plt.show()
			#cv2.imshow("Result", base2)
			#cv2.waitKey(0)
			
			print(f"{len(matches)} matches")
			

main()
