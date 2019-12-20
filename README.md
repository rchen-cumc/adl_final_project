# adl_final_project
Names: Tony Sun, Ray Chen
UNIs: tys2108, rc3179

1. To build the Docker image, do: `docker build -t <imageName>:<Tag> <Dockerfile path>`  
where Dockerfile path is the path to the Dockerfile in this repo. For example:  
`docker build -t LSTM_model:ver1 /Users/Tony/Documents/git/adl_final_project/Dockerfile`  
2. Once the Docker image is built, we want to run the Docker image. The data for this challenge is split into train and infer (see this GDrive: https://drive.google.com/drive/folders/1AU9npgYtEe44-fb-faitqQuer_JJ-Z3L for the data and the file structure). Mount the `/train/` and `/infer/` volumes using `-v` flag:  
`docker run -it LSTM_model:ver1 -v <path to train files>:/train/ -v <path to infer files>:/infer/ bash /app/train.sh`  
3. The above runs the training shell script. To run infer, mount the same folders, just run `infer.sh` instead:  
`docker run -it LSTM_model:ver1 -v <path to train files>:/train/ -v <path to infer files>:/infer/ bash /app/infer.sh`  
Note that it's possible to mount a `/scratch/` volume or a `/model/` volume, to save a copy of the models locally after training.
4. Without using Docker, there's also a copy of an end-to-end Colab-enabled notebook (see video of real-time execution); note that the GDrive files must be mounted, so clone the files on the shared drive into your own GDrive prior to running: https://colab.research.google.com/drive/1sqtpD2z2a0eu_6qz5r6FJhOrG2MNymI6
5. A copy of the results shown in the demo are here:  https://github.com/toekneesunshine/adl_final_project/blob/master/train_infer_all_in_one.ipynb
