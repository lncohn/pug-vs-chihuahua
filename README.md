#Deep Learning - Pugs vs. Chihuahuas.

This is the github repository for the Pug/Chihuahua classifier model and flask web app.
The project was inspired by Michelangelo D'Agostino's Strata 2016 talk and the VGG16 pretrained model developed for the ILSVRC-2014 competition. See www.lncohn.com for more information.

#Get the Data
The first step is to download the images from ImageNet.  If you have a university email address, you can download the images directly.  
If you do not, see the data folder for a list of url's of pug and chihuahua images. Write a script to download the images.  Then, crop all images to 224 x 224 pixels using imagemagick (brew install imagemagick). You will need imagemagick to use the flask app as well.  

#Make the Model
Make sure you have the following packages installed: scipy, numpy, scikit-image, requests, h5py, flask, keras, theano.   
Download the vgg16 weights, "vgg16.h5", available at the following github page https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3. Now run pugs_vs_chihuahuas.ipynb to build the model and evaluate_pugs_vs_chihuahuas.ipynb to evaluate the model. Important! This model takes quite a long to build on a macbook pro (but not an amazon ec2 GPU). 

#Run the Flask App
Copy the weights produced from building the model to the flask-app folder and run python pug_v_chi_app.py in the terminal.
Open your browser to the appropriate local ip address to view and use the flask app.  Try running the app on examples of pug/chihuahua 
mixes as well.