# Sanushi

## Approach
The overall approach is to:

 1. train an image classifier on the provided dataset
   * Since the dataset is quite small (only about 800 images in total), we should reuse pre-trained models. In our case we choose Inception v3.
 2. assuming the trained model achieves sufficient performance, use the prediction outputs to find images which the model finds difficult to classify (presumably because the dish has aspects of both classes)
 
## `kmodel` library and dependencies
In order to make this process easy to automate, I've developed various methods to help execute the approach described above, which can be found in folder `kmodel`. This is implemented using Keras. Basic use of this library can be seen in both the included jupyter notebook as in the standalone python script described below.

In order to use this library you'll need a python environment with a recent version of Keras, numpy, pandas, etc. Of course, having a GPU will speed up the training part.

## Jupyter notebook
To execute and inspect the approach you can execute the `Sanushi.ipynb` file on a local Jupyter server (or on [Google Collaboratory](https://colab.research.google.com/github/rdenaux/sashi/blob/master/Sanushi.ipynb)). The notebook provides some explanation of what is happening at each step of the process. Also, it provides an initial step for downloading the dataset and splitting it into a test and validation subset.

## Standalone execution
Since one of the requirements is full automation. I include `run.py` which provides a command-line script that can be executed for a given dataset. The main execution looks as follows:

```
python run.py -t <train_dir> -v <valid_dir> -i <img_dir> -p <json_params>
```

The `train_dir`, `valid_dir` and `img_dir` values should be a folder with subfolders named after the image classes and image files in those subfolders. Of course, all subfolders must refer to the same classes.
 * images in `train_dir` will be used for training the model
 * images in `valid_dir` will be used to validate the model (and store weights for the best performing models)
 * images in `img_dir` will be used to select candidate ambiguous images
 
Although the command-line interface requires three separate folders, if you are a bad person, you can pass the same folder to use the same dataset for training, validation and candidate selection.

 * The `json_params` is optional and should point to a json file with parameters required by the `kmodel` library. The repo includes `default_params.json` which provides an example of config values which seem to work OK. See `kmodel/kmodel.py` for a `sample_params` dict with comments describing (more or less) what each parameter means. If you understand the basic approach it should be clear what most parameters mean.
 
The output is a file called `sanushi_candidates.csv` which provides the top 20 most ambiguous images.

## Justifications
Many of the justifications are already described in the included notebook. But I repeat them here:

* The main code was derived from [this source file from github](https://github.com/aleksas/keras-fine-tune-inception/blob/master/fine_tune_inceptionv3.py).
* I use InceptionV3, since it provides a good balance of model size and accuracy. See [Keras application documentation](https://keras.io/applications/) for a table of other pre-trained models in case we have other priorities. E.g. if we want to be able to run inference on mobile phones, we could repeat this process with one of the smaller models; if we want to prioritise accuracy we could try `NASNetLarge`. In any case, the process is mostly the same.
* I've added a dropout after the top hidden layer to avoid overfitting (I didn't have time to play with this value to see if it actually is doing something).
* I've kept the final layer's activation as a softmax. I considered changing this to a sigmoid, to allow the model to predict both classes, but since the dataset is provided as a single class per image, I thought this would not make a big difference. If we had multi-label classification and more than two classes, this may still be a good idea.
* I've changed optimizer from rmsprop to adam, since that's what I tend to use most often and usually produces good results (again, I didn't have time to play around with hyperparameters, but the results seem good enough).
* Finetuning can result in overfitting, so I reduced the number of epochs for this phase to 5. It may be a good idea to use early stopping and to reload the weights to the best results before finding the candidates. Overfitting should be avoided as much as possible, especially if selecting candidate sanushis from the test set.

## Analysis and improvements
The method for finding candidate sanushis is simple, but should be sound. In some of my runs with the notebook, I did manage to find some photos of interesting mixes of sushi and sandwiches. 

On the other hand, there were also photos of things that were not a mixture; attribution studies could be used to analyse such images, e.g. with [visualizations](https://distill.pub/2017/feature-visualization/). These issues may also be due to the noisy dataset; e.g. I saw several images that were not really sandwiches, but rather just buns. I also noticed that some forms of sushi are already somewhat similar to sandwiches.

Of course, the main limitation right now is that this only works for pairs of dishes/snacks. I tried making the approach a bit more generalisable, it would be interesting to see how something similar could be made for multi-label images. As mentioned above, we would probably need to choose a different activation for the final layer and also revise how the distance between two classes is calculated.

