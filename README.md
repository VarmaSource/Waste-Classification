# Waste-Classification

##Chapter 1

Introduction

Waste classification has become a major concern for societies across the globe, as the volume of waste generated continues to increase. Manual waste classification is not only time-consuming but also often inaccurate. Machine learning offers a promising solution to this problem by enabling accurate and efficient waste classification. This technology uses algorithms and statistical models to analyze large datasets and identify patterns and trends. Once trained, machine learning models can classify waste with high accuracy, reducing the need for human intervention and improving overall waste management processes.
One of the key benefits of using machine learning for waste classification is its ability to learn from experience. As more waste is classified, the model becomes more accurate and efficient. This means that over time, the system will require less human input, reducing the likelihood of errors and improving overall efficiency. Additionally, machine learning can help identify the composition of waste, which can inform policy decisions and guide waste reduction efforts. Convolutional Neural Network is referred to as CNN. It is a kind of deep neural network widely used for tasks like natural language processing, image and video recognition, and other kinds of machine learning. Convolutional layers are used to identify features in the input image, and pooling layers are used to downsample the feature maps and lessen their dimensionality. CNNs can learn more intricate representations of an input image as they move up the layers thanks to this hierarchical architecture. A set of learnable filters are used by the convolutional layers in a CNN and are applied to local areas of the input image to create feature maps. These filters are capable of spotting details like edges, corners, and textures. The pooling layers then reduce the dimensionality of the feature maps by down-sampling them, typically by taking the maximum or average value over small regions. Overall, the use of machine learning for waste classification has enormous potential to revolutionize waste management processes. As the technology continues to evolve and become more sophisticated, it is likely that we will see even greater improvements in waste classification accuracy and efficiency.


##Chapter 2

Basic Concepts/ Literature Review

This section contains the basic concepts about the related tools and techniques used in this project. For research work, present the literature review in this section.

2.1 Tools Used:

2.1.1 Python: 
Python is a popular programming language used for machine learning projects due to its simplicity and extensive range of libraries and frameworks. Python libraries like TensorFlow, Keras, PyTorch, and Scikit-learn are commonly used for image classification tasks.

2.1.2 Jupyter Notebook:
Jupyter Notebook is an open-source web application that allows you to create and share documents containing code, equations, and visualizations. It is a popular tool for data analysis and machine learning projects.

2.1.3 Deep Learning Frameworks: 
Deep learning frameworks like TensorFlow, Keras, and PyTorch provide pre-built models and tools for building and training complex neural networks. They are commonly used in image classification tasks.

2.2 Techniques used:

2.2.1 Data collection:
The first step in building a machine learning model is to collect data. In the case of a waste classification system, this would involve gathering images of different types of waste, such as plastic bottles, aluminum cans, paper, and so on.

2.2.2 Data pre-processing:
Once you have collected your data, you will need to pre-process it to prepare it 
for use in your machine learning model. This might involve tasks such as resizing images to a consistent size, converting images to grayscale, and normalizing pixel values.

2.2.3 Training data preparation: 
Before you can train your machine learning model, you will need to split your data into training and validation sets. The training set is used to teach the model to recognize patterns in the data, while the validation set is used to evaluate the model's performance.

2.2.4 Model selection:
There are several types of machine learning models that can be used for image classification, including convolutional neural networks (CNNs), decision trees, and support vector machines (SVMs). CNNs are the most commonly used type of model for image classification tasks.

2.2.5 Model training:
Once you have selected a model, you will need to train it on your training data. This involves feeding the model input data and adjusting the model's weights and biases to minimize the difference between the model's predictions and the actual labels.

2.2.6 Hyperparameter Tuning:
Many machine learning models have hyperparameters that can be adjusted to improve performance. These might include the learning rate, the number of layers in the model, and the number of neurons in each layer. Hyperparameter tuning involves testing different combinations of hyperparameters to find the best ones for your model.

2.2.7 Model evaluation
Once you have trained your model, you will need to evaluate its performance on your validation set. This will give you an idea of how well your model is able to generalize to new, unseen data.

2.3 Literature Review

There have been several image classification research projects using neural networks in recent years, but there haven't been enough projects using computer vision and neural networks to classify trash.

In 2018, Rahmi et.al [1] developed a Deep learning CNN model to classify the waste into three classes of Recyclable, Residual, and Non-Recyclable classes. The author used a TrashNet dataset containing 2527 images that belong to various classes. He developed four Deep learning models Xception, MobileNet, Densely connected CNN, and Inception V4, in which his top accuracy models  were Inception V4 with 89% accuracy, DenseNet and MobileNet with 84% accuracy. Adam and Adadelta were the optimizers he used in his study. 

In 2016, Mindy et.al [2] have done a comparative analysis of SVM and CNN, where he concluded that SVM produces better results than CNN. They used SIFT algorithm for keypoint detection and 11-Levels in Neural Networks. SVM achieved a test accuracy of 66% while, CNN achieved a very low test accuracy of 22%. They classified the waste into three classes, Recyclable, Residual, and Non-Recyclable. 

In 2019, Victoria et.al [3] developed an automated system that captures images of waste items and classifies them into different categories such as paper, plastic, metal, and glass. The system uses a combination of image preprocessing techniques, feature extraction, and a support vector machine (SVM) classifier to achieve high classification accuracy. The authors tested the system using a dataset of waste images and achieved an overall classification accuracy of 92.17%. The proposed method has potential applications in waste management and recycling industries to improve waste sorting efficiency.

In 2020, Dipesh et.al [4] performed a comparative analysis of multiple CNN models for waste classification. They found that InceptionV3 had the highest accuracy and F1 score, while VGG16 had the highest precision and recall. The authors also analyzed the confusion matrix of each model to identify common misclassifications. The results showed that deep CNN models are effective for waste classification and could be useful in developing automated waste sorting systems.

In 2016, J.Donovan et.al [5] developed an Auto-Trash which was able to differentiate between compost and was recycled with Raspberry Pi, their system was developed using Google’s Tensorflow. The short-come of their system was that it was only able to differentiate compost materials. 

The authors in [6] proposed a deep learning framework using a two-stage detector to identify and classify garbage into seven categories with a level of accuracy up to 75%. The first stage manages litter localization without focusing on its class type with the help of EfficientDet-D2, and the second phase follows the classification of the identified waste into seven categories where the training of the classifier is undertaken by the un-labeled images in a semi-supervised manner. The scheme suggests a mobile application using Deep learning which can precisely identify the category of the waste item.

In 2019, Adedeji, O., & Wang, Z. (2019).  propose a waste classification system that employs deep learning convolutional neural network (CNN) to accurately classify waste into different categories. The system involves the collection of waste images, preprocessing and augmentation of the images, training of a CNN model, and classification of new waste images using the trained model. The authors conducted experiments using a dataset of waste images consisting of six different waste categories, and the results showed that the proposed system achieved an accuracy of 96.78% in waste classification.

In 2021, Majchrowska, Sylwia, et al. proposed a garbage classification application that uses deep learning techniques to classify waste materials into different categories. The proposed system involves the collection of waste images, preprocessing and feature extraction of the images, training of a deep neural network (DNN) model using the extracted features, and classification of new waste images into different categories using the trained DNN model.The authors suggest that the system can be further improved by using more advanced deep learning techniques and by increasing the size of the dataset used for training the DNN model.

In this paper, we will be comparing various CNN models for waste classification.


##Chapter 3

Problem Statement / Requirement Specifications

The process of waste classification is a critical task that is necessary for proper waste management. It involves the identification of waste materials based on their physical characteristics such as size, shape, texture, color, and composition. However, manual classification of waste materials is time-consuming, expensive, and prone to errors. Therefore, an automated waste classification system that can accurately and efficiently classify waste materials based on their images is required.

The proposed solution is a Waste Classification System using Machine Learning. The system aims to use machine learning algorithms to classify waste materials into four categories: Reusable, Hazardous, Residual, and Biodegradable. The system will provide a user interface that allows users to take an image of the item and classify it into one of the four categories. The system will require a training dataset of labeled waste material images to train the machine learning algorithm following which it will be able to recognize the waste material and categorize it.

This project will provide an efficient and accurate solution for waste material classification using machine learning. It will improve waste management by making waste classification more accessible, faster, and cost-effective. The proposed solution is expected to have a significant impact on the waste management industry, particularly in reducing the environmental impact of waste disposal, promoting recycling, and proper handling of hazardous waste materials.

3.1 Project Planning

1.Determine the features and functionality required for the Waste Classification System.
2.Define project scope and objectives.
3.Develop a project plan and timeline, including milestones and deadlines.
4.Identify and allocate resources, required.
5.Define project risks and develop a risk management plan.
6.Develop a project budget.
7.Obtain approval from Project Guide.

3.2 Project Analysis

The risk analysis for the project involves identifying potential project risks and their impact on the project. The risks may include technical risks such as the accuracy and efficiency of the machine learning algorithm, the quality of the training dataset, and the integration of the system with different hardware and software platforms. The risks may also include project management risks such as delays in project timelines, inadequate resource allocation, and budget overruns.

To mitigate the risks, the project team may implement risk management strategies such as regular project reviews, continuous testing and evaluation of the system, and contingency planning. Additionally, the team may establish clear communication channels with the  Project Guide to ensure that potential risks are identified and addressed in a timely manner.
3.3 System Design

3.3.1 Design Constraints

The system will be designed to work within the following constraints:

1.The system will be developed using open-source machine learning libraries and frameworks, including TensorFlow, Keras, and Python.
2.The system will be designed to work on a range of hardware and software platforms, including desktop and mobile devices.
3.The system will be designed to operate in various working environments, including industrial, commercial, and residential settings.
4.The system will be designed to handle a large volume of image data for training and classification purposes.

3.3.2 System Architecture

The system will have the following components:

1.Image Acquisition: This component will capture images of waste materials using a camera or other image capture device.
2.Image Preprocessing: This component will preprocess the images to improve their quality and reduce noise.
3.Feature Extraction: This component will extract relevant features from the preprocessed images.
4.Machine Learning Algorithm: This component will use a machine learning algorithm to learn from the features extracted from the training dataset.
5.Classification: This component will classify waste materials into four categories based on their images.
6.Feedback Mechanism: This component will allow users to report misclassifications or provide correct classifications of waste materials.


##Chapter 4

Implementation

In this machine learning project, we have implemented the model using various CNN models. A Convolutional Neural Network (CNN) is a type of deep learning algorithm that is particularly well-suited for image recognition and processing tasks. It is made up of multiple layers, including convolutional layers, pooling layers, and fully connected layers.
The convolutional layers are the key component of a CNN, where filters are applied to the input image to extract features such as edges, textures, and shapes. The output of the convolutional layers is then passed through pooling layers, which are used to down-sample the feature maps, reducing the spatial dimensions while retaining the most important information. The output of the pooling layers is then passed through one or more fully connected layers, which are used to make a prediction or classify the image.

4.1 Methodology
CNN models are widely used for image classification tasks, and there are several methods that can be used to build a CNN model for image classification.Here we have used Transfer learning, it involves using a pre-trained CNN model, such as VGG16, ResNet50V2, or InceptionV3, etc, that has been trained on a large dataset such as ImageNet. The pre-trained model can be fine-tuned on a smaller dataset specific to the classification task at hand. This method is particularly useful when the target dataset is small, and the model can leverage the learned features from the pre-trained model.

4.1.1 VGG16:
The convolutional neural network (CNN) architecture known as VGG16 was developed by the Visual Geometry Group at the University of Oxford in 2014. It is a deep learning model that has been widely used for picture classification tasks including object identification and recognition. The 16 convolutional and pooling layers of the VGG16 architecture are followed by three fully linked layers. The convolutional layers are designed to learn and extract features from the input picture, whilst the pooling layers downsample the feature maps to minimize the dimensionality of the data. Using the completely linked layers and the learned characteristics, the image is categorized.

4.1.2 MobileNetV3Small:
A variation of the MobileNet architecture, a series of neural network designs made for effective inference on mobile and embedded devices, is MobileNetV3Small. In 2019, Google released MobileNetV3Small, an updated version of the previous MobileNetV1 and MobileNet2 models. A thin neural network architecture called MobileNetV3Small is designed for low-latency and low-power applications. Compared to earlier MobileNet designs, it performs picture classification tasks with excellent accuracy while using fewer parameters and processes.

4.1.3 ResNet50V2:
A subset of the ResNet (Residual Network) architecture family of neural network topologies made for extremely deep networks is called ResNet50V2. The original ResNet50 model was updated in 2015 by Microsoft Research to become ResNet50V2. The ResNet design is built on the concept of residual connections, which let data flow straight from a block of layers' input to their output. This improves performance and lessens the issue of vanishing gradients that can arise in very deep neural networks, allowing the network to be trained to larger depths.

4.1.4 EfficientNetV2B0:
An adaptation of the EfficientNet architecture, a series of neural network topologies made for effective and precise picture categorization, is EfficientNetV2B0. Google unveiled EfficientNetV2B0 in 2021 as an enhanced version of the baseline EfficientNetB0 model. In comparison to earlier EfficientNet designs, EfficientNetV2B0 is intended to perform image classification tasks with high accuracy while utilizing fewer parameters and processing resources. It does so by combining a number of ways, such as effective channel and spatial attention processes, compound scaling, and enhanced training procedures.

4.1.5 InceptionV3: 
Google unveiled InceptionV3 in 2015, a deep neural network architecture for picture categorization. It's a modification of the original Inception architecture, which was created to deal with the issue of disappearing gradients in deep neural networks. Convolutional neural network InceptionV3 extracts information from the input picture by combining 1x1, 3x3, and 5x5 convolutional filters. A single feature map is created by concatenating the outputs of these simultaneous filters. As a result, the network is able to collect both low-level and high-level information from the input picture.

4.1.6 Xception:
In 2016, Google unveiled the deep neural network architecture called Xception for picture categorization. It is a development of the Inception architecture and is intended to increase the effectiveness and precision of image categorization models. The main concept of Xception is the utilization of depthwise separable convolutions, which divide the conventional convolution process into two independent operations: a depthwise convolution that filters individual input channels, followed by a pointwise convolution that combines the filtered channels. This strategy enables the network to minimize the number of parameters and operations needed, improving efficiency and reducing training time.

4.2 Testing and Verification Plan
The accuracy and results of a CNN model for image classification depend on several factors, including the dataset, model architecture, hyperparameters, and training procedure. Here the best model has turned out to be EfficientNetV2B0 with an accuracy score of 0.9312, it's the best we achieved in this project. Below is the overall models and their loss and accuracy scores:

Test ID	Test Model	Loss	Accuracy
T01	VGG16	0.3162	0.8969
T02	MobileNetV3Small	0.3124	0.8958
T03	ResNet50V2	0.2871	0.9177
T04	EfficientNetV2B0	0.1929	0.9312
T05	InceptionV3	0.3229	0.8969
T06	Xception	0.2556	0.9073

4.3 Result Analysis and Screenshots:
The result of CNN model for image classification are typically reported using various performance metrics, here we have shown using accuracy and loss for all the models that we have used:

4.4 Quality Assurance
Quality assurance is an essential step in developing a CNN model for image classification to ensure that it meets the desired quality standards. Here we have followed all the key aspects and guidelines such as Test Set Performance, Validation Set Performance, Visualization of the model, Documentation and most important one is Code Quality.


##Chapter 5

Conclusion and Future Scope

5.1	Conclusion
In this study, we have evaluated 6 pre-trained CNN models for the classification of waste. In our experiments, the best classification results were achieved using the EfficientNetV2 model with 93.12% test accuracy. Further improvement would likely be possible by fine-tuning some of the layers of the original model with a very small learning rate. The training dataset could be enlarged to achieve even more accurate results and prevent overfitting.

5.2    Future Scope
The improved model could be implemented in real life with IoT and computer vision. CV can identify and classify waste in real-time, and with the help of node-red we can toggle relays, which can segregate waste to their respective bins. This can result in efficient and automated waste sorting without the need of human labor.
