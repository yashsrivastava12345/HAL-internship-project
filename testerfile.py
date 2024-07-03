from deepface import DeepFace
import cv2
# Example usage
img_path = ("0.jpeg")
db_path = "D:\\New folder\\Face Detection\\DB"
img = ""
# Perform face recognition using DeepFace
try:
    dfs = DeepFace.find(img_path=img_path, db_path=db_path)
    print(dfs)
    for i in dfs:
        for j in i:
            img=dfs[0][j].to_string(index=False)
            break
        break
    print(img)
    lst=[i for i in img.split(".")]
    list1=[]
    count=0
    for i in lst:
        if(count%2==0):
            list1.append(i)
        else:
            continue
        count+=1

    for i in list1:
        imge = cv2.imread(i+".png")
        if imge is not None:
            cv2.imshow("My image", imge)
            cv2.waitKey(0)
        else:
            print("1Image not found!")
except ValueError:
    print("2Image not found!")
#print(dfs)
#print(dfs[[identity]])



