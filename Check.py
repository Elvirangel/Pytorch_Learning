import cv2
import numpy as np


# sobel
def Sobel(image):
    image = cv2.imread(path, 0)
    grad_x=cv2.Sobel(image,cv2.CV_16S,1,0)
    grad_y=cv2.Sobel(image,cv2.CV_16S,0,1)
    gradX=cv2.convertScaleAbs(grad_x)
    gradY=cv2.convertScaleAbs(grad_y)

    cv2.imshow("grad_x",gradX)
    cv2.namedWindow('grad_y', 0)
    cv2.imshow("grad_y",gradY)
    gradXY=cv2.addWeighted(gradX,0.5,gradY,0.5,0)
    cv2.imshow("gradXY",gradXY)
    cv2.imwrite("Sobel_Original.jpg",gradXY)
    newImage=cv2.imread(path,1)
    grad=np.stack((gradXY,gradXY,gradXY))
    print(grad.shape)
    grad=np.transpose(grad,[1,2,0])
    print(newImage.shape)
    newImage=newImage+grad
    cv2.imshow("newImage",newImage)
    cv2.imshow("image1",image)
    image=image+gradXY
    cv2.imshow("Image2",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Scharr(path):
    image = cv2.imread(path, 0)
    grad_x=cv2.Scharr(image,cv2.CV_16S,1,0)
    grad_y=cv2.Scharr(image,cv2.CV_16S,0,1)
    gradX=cv2.convertScaleAbs(grad_x)
    gradY=cv2.convertScaleAbs(grad_y)
    cv2.imshow("grad_x",gradX)
    cv2.namedWindow('grad_y', 0)
    cv2.imshow("grad_y",gradY)
    gradXY=cv2.addWeighted(gradX,0.5,gradY,0.5,0)
    cv2.imshow("gradXY",gradXY)
    newImage=cv2.imread(path,1)
    grad=np.stack((gradXY,gradXY,gradXY))
    print(grad.shape)
    grad=np.transpose(grad,[1,2,0])
    print(newImage.shape)
    newImage=newImage+grad
    cv2.imshow("newImage",newImage)
    cv2.imshow("image1",image)
    image=image+gradXY
    cv2.imshow("Image2",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Laplacian(path):
    image = cv2.imread(path, 0)
    dst=cv2.Laplacian(image,cv2.CV_16S)
    lpls=cv2.convertScaleAbs(dst)

    cv2.imshow("Laplacian",lpls)

    newImage=cv2.imread(path,1)
    grad=np.stack((lpls,lpls,lpls))
    print(grad.shape)
    grad=np.transpose(grad,[1,2,0])
    print(newImage.shape)
    newImage=newImage+grad
    cv2.imshow("newImage333",newImage)
    cv2.imshow("Original_image",image)
    image=image+lpls
    cv2.imshow("Image2",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__=="__main__":
    path = "29.jpg"

    Sobel(path)
    # Scharr(path)
    # Laplacian(path)