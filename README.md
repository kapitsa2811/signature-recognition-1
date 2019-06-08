# Tensorflow-Signature-Recognition
A siamese network implementation for signature recognition. A Resnet50 based 
siamese network is trained with triplet loss with the choice of online semi-hard negative
mining or hard negative mining. [[Blog for triplet loss]](https://github.com/omoindrot/tensorflow-triplet-loss) 
[[Blog for simaese]](https://towardsdatascience.com/siamese-network-triplet-loss-b4ca82c1aec8)
[[Trained Model]](https://drive.google.com/file/d/1MOReElVkaKo1zH_oMyTdxBGA5FimXgQq/view?usp=sharing)

The network is trained on open-source [signature dataset](https://cedar.buffalo.edu/NIJ/data/signatures.rar).
Please note that this code is not explicitly written for signature detection and can be used for any siamese task
such as Face-Recognition, writer recognition using handwritten text, etc. See Notes Section.

<img src="./images/Siamese.png">

## Dependencies

- Python 2.7 || 3.5 - 3.8
- Tensorflow == 1.12.0 (Should work on lower versions with minor changes)

The code is tested on :- Ubuntu 14.04 LTS with CPU architecture x86_64 + Nvidia Titan X 1070 + cuda9.0.

## Getting Started

### Training

First download the [dataset](https://cedar.buffalo.edu/NIJ/data/signatures.rar) 
and extract it to appropriate folder (divide dataset in two subsets train and val). The code is not data format dependent, 
thus can be used with any custom data. 

To run training, edit and run train.sh:
```bash
#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /handwritten-data/experiment_sign_semi_reg \
    --summary_dir /handwritten-data/experiment_sign_semi_reg/log/ \
    --mode train \
    --train_dir /handwritten-data/signatures/full_org \
    --val_dir /handwritten-data/signatures/val \
    --val_dataset_name kaggle_signature \
    --learning_rate 0.0001 \
    --loss semi-hard \
    --decay_step 50000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --loss_margin 0.5 \
    --max_iter 200000
```

For all available options, check main.py.


# Inference

I have implemented a spring boot Java application with react frontend to serve the model for inference, which is
available [here](https://github.com/rmalav15/siamese-tf-java). 
To use your model with Java app. Use graph_serialize_utils to convert (and visualize) tf model to (frozen) pb file.
and follow instruction on above repo.

# Notes

* The [pretrained model](https://drive.google.com/file/d/1MOReElVkaKo1zH_oMyTdxBGA5FimXgQq/view?usp=sharing) is not 
explicitly trained for signature fraud detection. 
* This code can be trained for any other siamese task by just providing appropriate train and val folder. 
No code change required.
* Other signature dataset
    * [SigComp2011](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2011_Signature_Verification_Competition_(SigComp2011))
    * [4NSigComp2012](http://www.iapr-tc11.org/mediawiki/index.php/ICFHR_2012_Signature_Verification_Competition_(4NSigComp2012))
    * [Kaggle](https://www.kaggle.com/divyanshrai/handwritten-signatures)
* Writer recognition dataset
    * [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
* face dataset
    * [CASIA-WebFace](https://github.com/happynear/AMSoftmax/issues/18)
    * [VGGFACE](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)