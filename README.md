# Usage

## Step1: Prepare the image folder in the following format

```
# Dataset struct
- Data
    -test
        - 1.jpg
        - 2.jpg
        - .......
        - ....
 
```

## Step2: change the config.py file

```
test_image_dir = "" ->give the image_dir path 
model_weights_path = "" ->give the model checkpoint path
results_file_path = "" ->give the path where OCR outputs will be stored

batch_size =   ->give the batch size 

```
# Step3: run the test module
```
python test.py
```

## Provided Four models trained on different types of MNIST Datasets in the "results" folder 

```
# Model's Folder Name  
    1.ResNet50_FineTune (Fine tuned on the last model with the extracted Images of actual scanned forms)
    2.resnet50_Mnist_withoutDegradation (Trained on MNIST BYCLASS DATASET with out adding noise)
    3.resnet50_Mnist_unbalanced_degradation (Trained on MNIST BYCLASS UNBALANCED DATASET with adding noises)
    4.ResNet50_Mnist_Balanced_degradation (Trained on MNIST BYCLASS BALANCED DATASET with adding noises)
        
```

## Accuracy of Four models on extracted images of actual scanned forms


```
1.ResNet50_FineTune -> 95%
2.resnet50_Mnist_withoutDegradation -> 57%
3.resnet50_Mnist_unbalanced_degradation -> 71%
4.ResNet50_Mnist_Balanced_degradation -> 65%
```