# Secondary Protein Structure Prediction

I worked on predicting the **secondary structure of proteins** from variable-length amino acid sequences. I wrote a detailed report on the project, which you can read on Medium: [Medium Article Link](your-medium-link).

The **inference API** for this project is fully containerized and available as a Docker image: [Docker Image Link](https://hub.docker.com/repository/docker/alokm7/protein2-api/general).  
The **project frontend** is live and can be accessed here: [Frontend Link](your-frontend-link).

---

## Problem Statement

I tackled the challenge of **secondary protein structure prediction**, which involves predicting both **SST8** and **SST3** labels for each residue in a protein sequence. 
The variable lengths of protein sequences and subtle differences between secondary structure classes make this a challenging problem.

---

## My Approach

- I implemented **Bi-RNNs** (including LSTM, GRU, and vanilla RNN) and **Transformers** from scratch for sequence modeling.  
- I experimented with **transfer learning**, using pretrained models to improve predictions.  
- I compared all models using **SST8 and SST3 benchmarks**, and the results are included in the repo as charts.

---

## Results

I included **model comparison charts** in this repository showing the performance of all approaches on SST8 and SST3 prediction. These charts helped me analyze which models perform better under different conditions.
![SST8 f1 scores comparing different models](src/plots/sst8_f1.png)

![SST3 validation scores with different models](src/plots/sst3_validation.png)

---

## Project Structure
src/
  ->data // contains all preprocessing and data loaders objects.
  ->models //contains all the model architectures.
      >preatraied // contains the prot_bert loading and classifier head of transfer learning.
      ->Scratch // contains model architectures of Bi-RNN , Bi-LSTM , Bi-GRU. 
      ->transformer // contains transformer architecture.
      ->training1 // training logic of "from scratch" models .
      ->training2 // training logic of pretrained model.
      ->validation // validation logic .
  ->plots // contains plotting logic and all plots .
  ->saved models // contain model checkpoints and saved weights.
  

#This project is completely for learning purposes and was part of my academic coursework for the B.S. degree in Data Science at IIT Madras.


