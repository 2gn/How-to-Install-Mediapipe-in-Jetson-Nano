# How to Install Mediapipe in Jetson Nano 2GB, 4GB, Jetson Xavier NX

Video Link - 
https://youtu.be/RAfkrusLnkM

Here, I Will show how to install MediaPipe in jetson nano and run an example program to make sure that it is working

The normal way to install mediapipe in Linux is not working 

So I have found a way to install mediapipe from source code, This way of installing is a little bit hard but it works very well in the jetsons.

I understand that the GitHub of mediapipe-bin has changed and you guys were getting errors while installing mediapipe , so I have fixed the error and changed the last step in my Github.

so now there should be no error and you can get mediapipe running in the jetson with no problem.

Thank you

So let us start this Tutorial 

In this tutorial, I am using a newly flashed jetson OS on my jetson nano

Step 1 - First setup the jetson nano then update it using - 

      sudo apt update

Now the system is updated and ready to install mediapipe

Step 2 -  Prerequisites and Dependencies

Now we need to install the Prerequisites and Dependencies for mediapipe and Tensorflow

1. Install system packages required by TensorFlow: -
      
          sudo apt-get update
          sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

2. Install and upgrade pip3:

          sudo apt-get install python3-pip
          sudo pip3 install -U pip testresources setuptools==49.6.0
      
3.Install the Python package dependencies:

                  sudo pip3 install -U --no-deps numpy==1.19.4 future==0.18.2 mock==3.0.5 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0 protobuf pybind11 cython pkgconfig
                  sudo env H5PY_SETUP_REQUIRES=0 pip3 install -U h5py==3.1.0

Step 3 - Now we need to install opencv-python:

     sudo apt-get install python3-opencv 
     sudo apt-get remove python3-opencv 

Step 4 - Increase swap for more swap ram 

https://github.com/JetsonHacksNano/installSwapfile.git

     git clone https://github.com/JetsonHacksNano/installSwapfile.git  
     cd installSwapfile

     ./installSwapfile.sh

Now all the Prerequisites and Dependencies are installed 

Now let’s get the mediapipe source code: 

     cd
     git clone https://github.com/google/mediapipe.git
     cd mediapipe

Now let's Install it from the source code:

    sudo apt-get install -y libopencv-core-dev  libopencv-highgui-dev libopencv-calib3d-dev libopencv-features2d-dev libopencv-imgproc-dev libopencv-video-dev
    sudo chmod 744 setup_opencv.sh
    ./setup_opencv.sh

Step 5 

    sudo pip3 install opencv_contrib_python
    sudo apt install curl


    so guys the medipipe-bin has changed so the installation is little  different from the video only this step has some changes , 
         Do as I say in this github 
        Now first you need to download this folder of medipipe-bin from this link - 
        
https://drive.google.com/file/d/1lHr9Krznst1ugLF_ElWGCNi_Y4AmEexx/view?usp=sharing

        After downloading the zip file unzip it by going to where it is been download in your jetson , mostly in the Downloads folder 
        
       cd Downloads
       
       sudo apt install unzip
        
       unzip mediapipe-bin.zip

Then run 

       cd mediapipe-bin
       
       sudo pip3 install numpy-1.19.4-cp36-none-manylinux2014_aarch64.whl mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl


       pip3 install dataclasses

  This is the new way to install mediapipe in jetson

    now continue with the video and complete the installation 
    Also do not forget to subscribe to my channel.
    
https://www.youtube.com/channel/UC0Nm0O9WIpnPzETvmGoI0jA/

Now all installation is completed , lets try to run a example program to make sure that it is working properly 

Download the file from my GitHub and try running it on your computer 
