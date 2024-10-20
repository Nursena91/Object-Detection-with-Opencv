import cv2
import os 
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image


'''
cv2.imread() #reads image
cv2.IMREAD_GRAYSCALE #0loads image in grayscale mode
cv2.IMREAD_COLOR #1loads a color image. Any transparency of image will be neglected
cv2.IMREAD_UNCHANGED #-1loads image as such alpha channel
print('imported')
'''

"""
#siyah beyaz renkte resmi okur ve matris halinde gösterir. (opencv'de bütün fotoğraflar matrise çevrilir ve her bir pixel matriste bir noktaya denk gelir.)
cb_image = cv2.imread("checkerboard.png", 0)
print(cb_image)

print("image size is: ", cb_image.shape)
print("data type of image is: ", cb_image.dtype)
"""

"""
#resmi olduğu renklerde okur. fakat opencv fotoğrafları rgb şeklinde değil bgr şeklinde okur. bu yüzden fotoğrafı göstermeden önce rgb formatına çeviririz.
cb_image2=cv2.imread("images.png", 1)
cb_image2_rgb = cv2.cvtColor(cb_image2, cv2.COLOR_BGR2RGB) #bgr'den rgb'ye çevirme
plt.imshow(cb_image2_rgb) #resmi yeni pencerede gösterir.
plt.show()
"""

"""
#resmi rgb renklerine ayırır.
img_bgr=cv2.imread("view.jpg", 1)
b, g, r=cv2.split(img_bgr)

plt.figure(figsize=[20,5])

plt.subplot(1,4,1)
plt.imshow(r, cmap='gray')
plt.title("Red Channel")

plt.subplot(1,4,2)
plt.imshow(g, cmap='gray')
plt.title("Green Channel")

plt.subplot(1,4,3)
plt.imshow(b, cmap='gray')
plt.title("Blue Channel")

#resmi tekrar rgb renklerinde birleştirir.
img_merg=cv2.merge((b,g,r))
plt.subplot(1,4,4)
plt.imshow(img_merg[:,:,::-1]) #parantez içindeki resmi bgr'den rgb'ye çevirmenin diğer yoludur.
plt.title("Merge Output")
plt.show()

cv2.imwrite("view.jpg", img_bgr)
"""

"""
#resmin istediğimiz bir kısmıyla oynama(rengini değiştirme)
cb_img = cv2.imread("checkerboard.png", 0)
cb_img_copy = cb_img.copy()
cb_img_copy[2,2] = 255
cb_img_copy[2,3] = 255
cb_img_copy[3,2] = 255
cb_img_copy[3,3] = 255

#cb_img_copy[2:3,2:3] = 255 #üstteki ile aynı görevi yapıyor
plt.imshow(cb_img_copy, cmap ='gray')
plt.show()
"""

"""
#resmin boyutunu değiştirme
cb_img = cv2.imread('view.jpg', 1)
#cb_img_resized = cv2.resize(cb_img, None, fx=2, fy=2)
#plt.imshow(cb_img_resized[:,:,::-1])

#desired_width = 800
#desired_height = 800
#dim = (desired_width, desired_height)
#cb_img_resized = cv2.resize(cb_img, dsize=dim, interpolation=cv2.INTER_AREA)
#plt.imshow(cb_img_resized[:,:,::-1])

desired_width = 400
aspect_ratio = desired_width/cb_img.shape[1]
desired_height = int(cb_img.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)
cb_img_resized = cv2.resize(cb_img, dsize = dim, interpolation = cv2.INTER_AREA)
plt.imshow(cb_img_resized[:,:,::-1])
plt.show()
"""

"""
#resmi kırpma
cb_img = cv2.imread('view.jpg', 1)
cb_img_cropped = cb_img[100:300, 200:500]
plt.imshow(cb_img_cropped[:,:,::-1])
plt.show()
"""

"""
#resmi çevirme
cb_img = cv2.imread('view.jpg', 1)
cb_img_rgb = cb_img[:,:,::-1]
cb_img_flip_hor = cv2.flip(cb_img_rgb, 1)
cb_img_flip_ver = cv2.flip(cb_img_rgb, 0)
cb_img_flip_both = cv2.flip(cb_img_rgb, -1)

plt.figure(figsize=[18,5])
plt.subplot(1,4,1);plt.imshow(cb_img_flip_hor);plt.title("Horizontal flip")
plt.subplot(1,4,2);plt.imshow(cb_img_flip_ver);plt.title("Vertical flip")
plt.subplot(1,4,3);plt.imshow(cb_img_flip_both);plt.title("Both flip")
plt.subplot(1,4,4);plt.imshow(cb_img_rgb);plt.title("Original")
plt.show()
"""

"""
#resme şekil çizme
cb_img_line = cv2.imread('images.png', 1)
cv2.circle(cb_img_line, (115,115) , 80 , (20,30,20) , thickness=5,lineType=cv2.LINE_AA)
cv2.line(cb_img_line, (0,0) , (200,200) , (0,255,60), thickness=5, lineType=cv2.LINE_AA)
cv2.rectangle(cb_img_line, (1,1) , (100,100) , (150,50,200) , thickness=5 , lineType=cv2.LINE_AA)
text = "Hello Coca-Cola"
fontScale = 0.7
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (255,255,255)
fontThickness = 1
cv2.putText(cb_img_line, text, (120,10) , fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)
plt.imshow(cb_img_line[:,:,::-1])
plt.show()
"""

"""
#videoyu açma (ben de videoyu fotoğraf şeklinde açmıştı sorunu çözemedim)
source = 'dilalım.MP4' #videonun ismine takılma kjfkjfhvkjdfvh
cap = cv2.VideoCapture(source)
if(cap.isOpened() == False):
    print("Error opening video stream or file")
ret,frame = cap.read()
plt.imshow(frame[:,:,::-1])
plt.show()
"""

"""
#videoyu kaydetme
video = 'dilalım.MP4'
cap = cv2.VideoCapture(video)
while(cap.isOpened()):
    ret,frame = cap.read()
    if ret ==True:
        video.write(frame)
    else:
        break
"""

"""
#kamera açıp video kaydetme
# VideoWriter için codec ve çıkış dosyası ayarları (her video tipi için codec farklı olabilir. uygun olanı bulmak gerekiyor. ben chat gpt ye sorup bulmuştum.)
out_mp4 = cv2.VideoWriter('C:/Users/YourUsername/Desktop/deneme1.MP4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
alive = True
win_name = 'Acessing Camera'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
source = cv2.VideoCapture(0)

while alive:
    ret, frame = source.read()
    if not ret:
        break
    cv2.imshow(win_name, frame)
    out_mp4.write(frame)    
    #plt.show() OpenCV ile video işleme sırasında gerekli değil. 
    if cv2.waitKey(1) == ord('q'):
        alive = False
# Kaynakları serbest bırak
source.release()
out_mp4.release()
cv2.destroyAllWindows()
print("Current working directory:", os.getcwd()) #dosyayı nereye kaydettiğini bulamıyodum o işe yarıyo kvfbvxjkfvbxkfv
"""

"""
#parlaklık ayarlama
cb_image_bgr=cv2.imread("Landscape.jpg", 1)
cb_image=cv2.cvtColor(cb_image_bgr, cv2.COLOR_BGR2RGB)
matrix=np.ones(cb_image.shape, dtype = "uint8")*100
cb_image_brighter = cv2.add(cb_image, matrix)
cb_image_darker = cv2.subtract(cb_image, matrix)
plt.figure(figsize=[14,5])
plt.subplot(131); plt.imshow(cb_image_darker); plt.title("darker")
plt.subplot(132); plt.imshow(cb_image_brighter); plt.title("brigther")
plt.subplot(133); plt.imshow(cb_image); plt.title("original")
plt.show()
"""

"""
#kontrast ayarlama
cb_image_bgr=cv2.imread("Landscape.jpg", 1)
cb_image=cv2.cvtColor(cb_image_bgr, cv2.COLOR_BGR2RGB)
matrix1=np.ones(cb_image.shape)*.7
matrix2=np.ones(cb_image.shape)*2
cb_image_lower = np.uint8(cv2.multiply(np.float64(cb_image), matrix1))
cb_image_higher = np.uint8(np.clip(cv2.multiply(np.float64(cb_image), matrix2),0,255))
plt.figure(figsize=[14,5])
plt.subplot(131); plt.imshow(cb_image_lower); plt.title("lower contrast")
plt.subplot(132); plt.imshow(cb_image_higher); plt.title("higher contrast")
plt.subplot(133); plt.imshow(cb_image); plt.title("original")
plt.show()
"""

"""
#bulanıklaştırma
cb_image = cv2.imread("Landscape.jpg", 1)
cb_image_blur = cv2.blur(cb_image,(50,50))
plt.figure(figsize=[15,5])
plt.subplot(121); plt.imshow(cb_image[:,:,::-1]); plt.title("original")
plt.subplot(122); plt.imshow(cb_image_blur[:,:,::-1]); plt.title("blurred")
plt.show()
"""

"""
#ana hatları gösterme
cb_image_bgr = cv2.imread("Landscape.jpg", 1)
cb_image = cv2.cvtColor(cb_image_bgr, cv2.COLOR_BGR2RGB)
cb_image_canny= cv2.Canny(cb_image, 200, 200)
plt.figure(figsize=[15,5])
plt.subplot(121); plt.imshow(cb_image); plt.title("original")
plt.subplot(122); plt.imshow(cb_image_canny); plt.title("edged")
plt.show()
"""

"""
#fotoğraftaki köşe noktaları algılama
cb_image_bgr = cv2.imread('person.jpg', 1)
cb_image_gray = cv2.cvtColor(cb_image_bgr, cv2.COLOR_BGR2GRAY)
feature_params = dict(maxCorners = 1000,
                      qualityLevel = 0.00001,
                      minDistance = 5,
                      blockSize = 9)
corners = cv2.goodFeaturesToTrack(cb_image_gray, **feature_params)
corners = np.intp(corners)
if corners is not None:
    for x, y in np.float32(corners).reshape(-1,2):
        cv2.circle(cb_image_bgr, (int(x),int(y)),5, (0,255,0) ,1)
else:
    print("No corners found")
cb_image_rgb = cv2.cvtColor(cb_image_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(cb_image_rgb)
plt.show()
"""

"""
cb_img = cv2.imread('j.png')
assert cb_img is not None , "amınakodumun hatası"
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(cb_img, kernel, iterations= 1)
diolation = cv2.dilate(cb_img, kernel, iterations=1)
opening = cv2.morphologyEx(cb_img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(cb_img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
plt.show()
"""

"""
cb_img = cv2.imread('thr.jpg',1)
img = cv2.cvtColor(cb_img, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 100, 255, cv2.THRESH_TOZERO_INV)
thresh6 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,7)
thresh7 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,7)
titles = ['Original', 'Binary', 'Binary_Inv', 'Trunc', 'Tozero', 'Tozero_Inv', 'Adaptive_Mean_C', 'Adaptive_Gaussian']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]
plt.figure(figsize = [15,5])
for i in range(8):
    plt.subplot(2,4,i+1),plt.imshow(images[i], cmap='gray'),plt.title(titles[i])
plt.show()
"""

"""
cb_img = cv2.imread('shapes.webp')
img = cv2.cvtColor(cb_img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(img, 30, 200)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(img, contours, 2, (0, 255, 0), 5)
plt.imshow(img)
plt.show()
"""

"""
#step1: read the template and scanned version
img1 = cv2.imread('painting_original.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('painting_scanned.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#step2: find keypoints in both images
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

MAX_NUM_FEATURES= 500
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

img1_display = cv2.drawKeypoints(img1, keypoints1, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_display = cv2.drawKeypoints(img2, keypoints2, outImage=np.array([]), color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1), plt.imshow(img1_display), plt.title("Keypoints in Scanned Image")
#plt.subplot(1, 2, 2), plt.imshow(img2_display), plt.title("Keypoints in Original Image")
#plt.show()

#step3: match the keypoints in both images
#match features
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1,descriptors2, None)
#sort matches by score
matches = sorted(matches, key = lambda x: x.distance, reverse=False)
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]
im_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None)
#plt.figure(figsize=(10, 5))
#plt.imshow(im_matches), plt.title("Suck it")
#plt.show()


#step4: find homography
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
    points1[i,:] = keypoints1[match.queryIdx].pt
    points2[i,:] = keypoints2[match.trainIdx].pt
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

#step5: use homography to warp image
height, width, channels = img1.shape 
img2_reg = cv2.warpPerspective(img2, h, (width, height))
plt.figure(figsize=(10, 5))
plt.subplot(121);plt.imshow(img1), plt.title("original")
plt.subplot(122);plt.imshow(img2_reg), plt.title("scanned form")
plt.show()
"""

"""
imageFiles = glob.glob("Room/*")
imageFiles.sort()

images = []
for filename in imageFiles:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)
if status == 0:
    plt.figure(figsize=[30,10])
    plt.imshow(result)
plt.show()
"""



