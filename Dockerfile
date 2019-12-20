FROM spellrun/tensorflow2-cpu
#FROM tensorflow/tensorflow:nightly-gpu-py3

# Install dependencies
RUN pip install pandas \
  numpy \
  datetime \
  matplotlib \
  tensorflow \
  tqdm

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY ./train.py .
COPY ./infer.py .
COPY ./preprocess_fn.py .
COPY ./common_functions.py .
COPY ./OMOP_usefule_columns.csv .
COPY ./condition_occurrence_concepts.csv .
COPY ./procedure_occurrence_concepts.csv .
COPY ./drug_exposure_concepts.csv .

# For testing in Docker
COPY ./model_train_only.py .
COPY ./model_train_only.sh .
COPY ./model_infer_only.py .
COPY ./model_infer_only.sh .

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY ./train.sh .
COPY ./infer.sh .

# Add executable permission to Bash scripts
RUN chmod +x infer.sh
