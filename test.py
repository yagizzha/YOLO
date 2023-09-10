from ultralytics import YOLO
import cv2
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from ultralytics.yolo.engine.results import Results
import math
import numpy as np
from flask import Flask,make_response,jsonify
from flask import request
import json
import dns.resolver
from time import time
import torch
from time import sleep
import requests


dns.resolver.default_resolver=dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers=['8.8.8.8']

app = Flask(__name__)


def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def determine_corners(points):
    # Sort points by x-coordinate
    points_sorted = sorted(points, key=lambda p: p[0])
    # Get left and right points
    left_points = sorted(points_sorted[:2], key=lambda p: p[1])
    right_points = sorted(points_sorted[2:], key=lambda p: p[1])
    return [left_points[0], right_points[0], right_points[1], left_points[1]]

    # Return corners
    #    "top_left": left_points[0],
    #    "bottom_left": left_points[1],
    #    "top_right": right_points[0],
    #    "bottom_right": right_points[1]

def warp_image(img1, img2, points):
    # Define the target coordinates for the transformation
    target_coords = np.array([(0, 0), (img2.shape[1], 0), (img2.shape[1], img2.shape[0]), (0, img2.shape[0])], dtype=np.float32)

    # Convert the input points to a numpy array of floating point values
    input_coords = np.array(points, dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(target_coords , input_coords)

    
    whiteimg = np.ones((img2.shape[0], img2.shape[1], 3), dtype=np.uint8) * 255
    resultWhite = cv2.warpPerspective(whiteimg, matrix, (img1.shape[1], img1.shape[0]))

    # Warp the second image based on the transformation matrix
    result = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))

    # Merge the warped second image with the first image
    for i in range(len(resultWhite)):
        for j in range(len(resultWhite[0])):
            if resultWhite[i][j][0]==255 and resultWhite[i][j][1]==255 and resultWhite[i][j][2]==255:
                img1[i][j]=result[i][j]

    # Return the merged image
    return img1

def warp_image_alpha(img1, img2, points):
    # Convert images to 4-channel (BGRA)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    # Define the target coordinates for the transformation
    target_coords = np.array([(0, 0), (img2.shape[1], 0), (img2.shape[1], img2.shape[0]), (0, img2.shape[0])], dtype=np.float32)

    # Convert the input points to a numpy array of floating point values
    input_coords = np.array(points, dtype=np.float32)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(target_coords, input_coords)

    alpha_img = np.ones((img2.shape[0], img2.shape[1], 4), dtype=np.uint8) * 255
    result_alpha = cv2.warpPerspective(alpha_img, matrix, (img1.shape[1], img1.shape[0]))

    # Warp the second image based on the transformation matrix
    result = cv2.warpPerspective(img2, matrix, (img1.shape[1], img1.shape[0]))

    # Set alpha to half transparency for the warped image
    result[:, :, 3] = (result_alpha[:, :, 3] * 0.5).astype(np.uint8)  # 0.5 for half transparency

    # Composite the images
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result_alpha[i][j][3] > 0:  # Check the alpha value
                alpha = result[i][j][3] / 255.0
                img1[i][j] = ((1 - alpha) * img1[i][j] + alpha * result[i][j]).astype(np.uint8)

    # Return the merged image
    return img1

testPlateReplacement=cv2.imread("testplate2.png")

model = YOLO("bestEU.pt")

def find_corners(image):
    if len(image.shape) > 2:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = image.copy()
    _, binary = cv2.threshold(grayscale, 10, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = min(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour)
    corners = [tuple(point[0]) for point in hull]

    return corners

def mark_points_on_image(image, points, color=(0, 0, 255), radius=15, thickness=-1):
    marked_image = image.copy()
    
    for point in points:
        marked_image = cv2.circle(marked_image, tuple(point), radius, color, thickness)
    
    return marked_image
def len_print(arr):
    dimensions = len(arr.shape)

    if dimensions == 2:
        y_length, x_length = arr.shape
        print(f"X Length: {x_length}")
        print(f"Y Length: {y_length}")
    elif dimensions == 3:
        depth, y_length, x_length = arr.shape
        print(f"Depth: {depth}")
        print(f"X Length: {x_length}")
        print(f"Y Length: {y_length}")
    else:
        print("Shape of array:", arr.shape)

def resize_with_whitespace(img):
    # Determine the shape of the image
    height, width, _ = img.shape

    # Determine the larger dimension
    max_dim = max(height, width)

    # Create a blank white image with the new dimensions
    result = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255

    # Since we are adding to the right and bottom, the offsets for top and left are 0
    y_offset = 0
    x_offset = 0

    # Place the original image onto the blank one
    result[y_offset:y_offset+height, x_offset:x_offset+width] = img

    return result


def show_detected_corners(img, threshold=150, nonmax_suppression=True, type_9_12=12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize FAST object with desired values
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression, type=type_9_12)

    # Find and draw the keypoints
    keypoints = fast.detect(gray, None)
    cv2.drawKeypoints(img, keypoints, img, color=(0, 0, 255))

    #cv2.imshow('FAST Corners', img)
    cv2.waitKey(0)


def detect(screen,bgr,cnf=0.20):
    timer=time()
    screencopy=screen.copy()
    screen=resize_with_whitespace(screen)
    #show_detected_corners(screen)
    print("time to copy",time()-timer)
    timer=time()
    results = model.predict(source=screen,show=False,hide_labels=True,line_thickness=0,conf=cnf)
    print("time to show",time()-timer)
    timer=time()
    #print(results)
    for result in results:
        #print(type(result),result)
        #print(type(result.boxes))
        #print(result.masks.masks.cpu().numpy()[0])


        #mask=result.masks.masks.cpu().numpy()
        #x = cv2.resize(mask[0], (len(screen[0]),len(screen)))
        #len_print(mask[0])
        #len_print(x)

        x = result.masks.masks[0].cpu().numpy()
        corners = cv2.goodFeaturesToTrack(x, 4, 0.1 , 30)  # get 4 corners
        pix=0
        for corner in corners:  # plot the corners on the original image
            x,y = corner.ravel()
            cv2.circle(screen,(int(x*len(screen)/640), int(y*len(screen)/640)),5,(255, pix, 0),-1)
            pix+=50
        #cv2.imshow("center_marked",screen)



        masks = result.masks.masks
        boxes = result.boxes.boxes
        # extract classes
        clss = boxes[:, 5]
        # get indices of results where class is 0 (people in COCO)
        people_indices = torch.where(clss == 0)
        # use these indices to extract the relevant masks
        people_masks = masks[people_indices]
        # scale for visualizing results
        people_mask = (torch.any(people_masks, dim=0).int() * 255).cpu().numpy()
        people_mask = people_mask.astype(np.uint8)

        x= people_mask


        #cv2.imshow("maskalone1",people_mask)
        #cv2.imshow("maskalone2",x)

        x = cv2.resize(x, (len(screen[0]),len(screen)))
        #print(x.dtype)
        #cv2.imshow("base",x)

        # Display the new grayscale image
        #cv2.imshow("new_image", x)


        #binary = x.copy().astype(np.uint8)

        #cv2.imshow("img_gpt",img_gpt)

        #x = cv2.GaussianBlur( x, (9,9), 0)
        #x = cv2.addWeighted( x, 1.5, x, -0.5, 0)

        """
        x = x.astype(np.uint8)
        for i in range(len(x)):
            for j in range(len(x[0])):
                if x[i][j]!=0 and x[i][j]!=255:
                    if x[i][j]<50:
                        x[i][j]=0
                    else:
                        x[i][j]=255
        pointsx=[]
        pointsy=[]
        for i in range(len(x)-1):
            for j in range(len(x[0])-1):
                if x[i+1][j]==0 and x[i][j]==255:
                    pointsx.append(j)
                    pointsy.append(i)
    
        pointx=sum(pointsx)/len(pointsx)
        pointy=sum(pointsy)/len(pointsy)
        """



        #cv2.imshow("postblur",x)

        return transform_image_color(screencopy,x,bgr)
        contours,_ = cv2.findContours(x, 1, 1)
        #print("len",len(contours),contours)
        screen=cv2.drawContours(screen,[contours[0]],-1,(0,255,0),2)
        #cv2.imshow("test232",screen)

        rect = cv2.minAreaRect(contours[0])

        box=cv2.boxPoints(rect)
        box=np.int0(box)

        cv2.drawContours(screen,[box],0,(0,0,255),2)
        #cv2.imshow("test233",screen)
        #print(box)
        box=determine_corners(box)
        #print("post",box)
        for corner in box:  # plot the corners on the original image
            #print("XY",corner[0],corner[1])
            cv2.circle(screen,(int(corner[0]), int(corner[1])),5,(0, 255, 255),-1)

        
        #cv2.imshow("overlayed",warp_image(screencopy,testPlateReplacement,box))
        print("time to finish",time()-timer)
        return warp_image(screencopy,testPlateReplacement,box)
    #cv2.imshow("test231",screen)
    #cv2.waitKey(1)

def average_array(arr):
    return sum(arr) / len(arr)

def transform_image_color(original, bw, bgr, threshold=240):
    (b,g,r)=bgr
    for i in range(len(original)):
        for j in range(len(original[i])):
            if bw[i][j]>threshold:
                original[i][j][0]=b
                original[i][j][1]=g
                original[i][j][2]=r

    return original

def hex_to_bgr(hex_color: str):
    # Remove '#' prefix if it exists
    hex_color = hex_color.lstrip('#')
    
    # Convert HEX to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Return as BGR
    return (b, g, r)


@app.route('/health-check',methods=['GET'])
def getHealthCheck():
    response=jsonify({"status":"healthy"})
    return response

@app.route('/turntopng',methods=['POST'])
def returnEndProduct():
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img=detect(img)
    #cv2.imwrite("recieved.png",img)
    retval, buffer = cv2.imencode('.png', img)
    response=make_response(buffer.tobytes())
    #print(response)
    return response


@app.route('/fill',methods=['POST'])
def fillPlates():
    jsonfile = request.json
    imageLinks = jsonfile["imageLinks"]
    colorCode = request.json['colorCode']
    albumId = request.json['albumId']
    uploadedImageUrls = []
    bgr=hex_to_bgr(colorCode)
    for imageLink in imageLinks:
        try:
            response = requests.get(imageLink)
            print(imageLink)
            image_np_array = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
            img=detect(image,bgr)
            is_success, image_buffer = cv2.imencode(".png", img)
            print("is-suc:", is_success, "try on url:", ("http://www.liplate.app/api/upload-image-response/" + albumId))
            if is_success:
                requests.post(
                    "http://www.liplate.app/api/upload-image-response/" + albumId,
                    data=image_buffer.tobytes(),
                    headers={'Content-Type': 'image/png'}
                )
        except Exception as e:
            print(e)
    requests.get(
        "https://www.liplate.app/api/finalize-album-creation/" + albumId,
        headers={'Content-Type': 'application/json'}
    )
    return jsonify({"message": "Finished the job for the album with the id:" + albumId})


if __name__=='__main__':
    for rule in app.url_map.iter_rules():
        print(f"Endpoint: {rule.endpoint} - Route: {rule.rule} - Methods: {' | '.join(rule.methods)}")
    app.run(host='0.0.0.0',port=2999)

    pass
