# Keras_Flask_classifier

 ## Requirement
 ```
 flask == 1.1.1
 tensorflow == 2.1.0
 Keras == 2.3.1
 Numpy == 1.18.1
 opencv == 4.1.0
 ```

## How to use this app

1. Type this  command in Terminal
```
$ cd Keras_Flask_classifier
```
2. Type this command in Terminal
```
$ python3 run.py
```

3. Put the link to your web browser 
*The link will show up when you execute the procedure 2
```
http://127.0.0.1:5000/
```
![enter image description here](https://user-images.githubusercontent.com/45162150/77249586-8bcadf00-6c85-11ea-8825-9f3b61a73aa7.png)
4. Input an image of woman
![enter image description here](https://user-images.githubusercontent.com/45162150/76144896-f4cc2780-60c7-11ea-876a-e43459ba8100.png)
 
5. The app will tell you whether the woman is my type or not
(It sometimes fail to detect the face in the uploaded image)
![enter image description here](https://user-images.githubusercontent.com/45162150/77249611-c6cd1280-6c85-11ea-96ff-5bb79812c963.png)

**Caution!**
The face detection model used in this app doesn't recognize leaned faces.
![enter image description here](https://user-images.githubusercontent.com/45162150/77249687-4eb31c80-6c86-11ea-925e-64051b5cdf1b.png)


## Reference
1. https://github.com/nishipy/obama-smalling-flask
(Actually, I borrowed most of the code in the repository above)

2. https://qiita.com/chilitreat/items/a1dab6c6b5ba088123e0

