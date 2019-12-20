# adl_final_project
Names: Tony Sun, Ray Chen
UNIs: tys2108, rc3179

To build the Docker image, do:
`docker build -t <imageName>:<Tag> <Dockerfile path>`
where Dockerfile path is the path to the Dockerfile in this repo. For example:
`docker build -t LSTM_model:ver1 /Users/Tony/Documents/git/adl_final_project/Dockerfile`

Once the Docker image is built, we want to run the Docker image. The data for this challenge is split into train and infer (see this GDrive: https://drive.google.com/ for the data). Mount the `/train/` and `/infer/` volumes using `-v` flag:
`docker run -it LSTM_model:ver1 -v <path to train files>:/train/ -v <path to infer files>:/infer/ bash /app/train.sh`
The above runs the training shell script. To run infer, mount the same folders, just run `infer.sh` instead:
`docker run -it LSTM_model:ver1 -v <path to train files>:/train/ -v <path to infer files>:/infer/ bash /app/infer.sh`
Note that it's possible to mount a `/scratch/` volume or a `/model/` volume, to save a copy of the models locally after training.
